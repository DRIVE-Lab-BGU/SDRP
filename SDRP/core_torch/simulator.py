"""Torch-based simulator mirroring the interface of JaxRDDLSimulator."""

import time
from typing import Callable, Dict, Optional, Union, Any

import numpy as np
import torch

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.debug.exception import (
    RDDLActionPreconditionNotSatisfiedError,
    RDDLInvalidExpressionError,
    RDDLStateInvariantNotSatisfiedError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.parser.expr import Value
from pyRDDLGym.core.simulator import RDDLSimulator

try:
    from .compiler import TorchRDDLCompiler  # type: ignore
except Exception:  # pragma: no cover - compiler will be provided later
    TorchRDDLCompiler = None  # type: ignore

Args = Dict[str, Union[np.ndarray, torch.Tensor, Value, float, int, bool]]




def _tree_map(fn: Callable[[Any], Any], tree: Any) -> Any:
    """Apply `fn` recursively to values living inside nested containers."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for (k, v) in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, v) for v in tree)
    return fn(tree)


class TorchRDDLSimulator(RDDLSimulator):


    def __init__(self, rddl: RDDLLiftedModel,
                 generator: Optional[torch.Generator]=None,
                 raise_error: bool=True,
                 logger: Optional[Logger]=None,
                 keep_tensors: bool=False,
                 objects_as_strings: bool=True,
                 python_functions: Optional[Dict[str, Callable]]=None,
                 logic: Optional[object]=None,
                 compiler_factory: Optional[Callable[..., Any]]=None,
                 **compiler_args) -> None:
    
        
        
        """Creates a simulator for the given RDDL model with torch backend.

        This mirrors the public API of :class:`pyRDDLGym_jax.core.simulator.JaxRDDLSimulator`.
        """
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(round(time.time() * 1000))
        self.key = generator
        self.raise_error = raise_error
        #####
        self.logic = logic
        self.compiler_factory = compiler_factory or TorchRDDLCompiler
        self.compiler_class = None
        self._compiled = None
        self.compiler_args = compiler_args
        ##### 

        super(TorchRDDLSimulator, self).__init__(
            rddl, logger=logger,
            keep_tensors=keep_tensors, objects_as_strings=objects_as_strings,
            python_functions=python_functions)

    def seed(self, seed: int) -> None:
        super(TorchRDDLSimulator, self).seed(seed)
        key_device = getattr(self.key, 'device', torch.device('cpu'))
        if isinstance(key_device, torch.device):
            key_device = key_device.type
        self.key = torch.Generator(device=key_device)
        self.key.manual_seed(seed)

    def _compile(self):
        if self.compiler_factory is None:
            raise ImportError(
                'TorchRDDLCompiler is not available. '
                'Provide a compiler_factory argument pointing to a torch compiler.')

        rddl = self.rddl
        compiler_kwargs = dict(self.compiler_args)
        if 'logic' not in compiler_kwargs and self.logic is not None:
            compiler_kwargs['logic'] = self.logic
        compiled = self.compiler_factory(
            rddl,
            logger=self.logger,
            python_functions=self.python_functions,
            **compiler_kwargs
        )
        compiled.compile(log_expr=True, log_jax_expr=False, heading='SIMULATION MODEL')
        ###
        self._compiled = compiled
        self.compiler_class = compiled.__class__
        #### 
        self.init_values = compiled.init_values
        self.levels = compiled.levels
        self.traced = compiled.traced

        self.invariants = _tree_map(self._wrap_callable, compiled.invariants)
        self.preconds = _tree_map(self._wrap_callable, compiled.preconditions)
        self.terminals = _tree_map(self._wrap_callable, compiled.terminations)
        self.reward = self._wrap_callable(compiled.reward)
        torch_cpfs = _tree_map(self._wrap_callable, compiled.cpfs)
        self.model_params = compiled.model_params

        ########## #### #### #### #### #### #### #### #### #### 
        torch_types = getattr(compiled, 'TORCH_TYPES', getattr(compiled, 'JAX_TYPES', {}))
        torch_int = getattr(compiled, 'INT', torch.int32)
        ######### #### #### #### #### #### #### #### #### #### 
        self.cpfs = []
        for cpfs in self.levels.values():
            for cpf in cpfs:
                expr = torch_cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = torch_types.get(prange, torch_int) if isinstance(torch_types, dict) else torch_int
                self.cpfs.append((cpf, expr, dtype))

        # Initialize all fluent and non-fluent values
        self.subs = self.init_values.copy()
        self.state = None
        self.noop_actions = {
            var: values for (var, values) in self.init_values.items()
            if rddl.variable_types[var] == 'action-fluent'
        }
        # this use function from model
        self.grounded_noop_actions = rddl.ground_vars_with_values(self.noop_actions)
        self.grounded_action_ranges = rddl.ground_vars_with_value(rddl.action_ranges)
        self._pomdp = bool(rddl.observ_fluents)
        # cached for performance 
        self.invariant_names = [f'Invariant {i}' for i in range(len(rddl.invariants))]
        self.precond_names = [f'Precondition {i}' for i in range(len(rddl.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(rddl.terminations))]
    
    
    #### i dont see it in the oroginal simulator #### 
    def _wrap_callable(self, func: Optional[Callable]):
        if func is None:
            return None

        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped
    #### #### #### #### #### #### #### #### #### #### #### #### #### 

    def handle_error_code(self, error: int, msg: str) -> None:
        if self.raise_error and error:
            compiler = self.compiler_class
            if compiler is not None and hasattr(compiler, 'get_error_messages'):
                errors = compiler.get_error_messages(error)  # type: ignore
            else:
                errors = []
            if errors:
                message = f'Internal error in evaluation of {msg}:\n'
                errors = '\n'.join(f'{i + 1}. {s}' for (i, s) in enumerate(errors))
                raise RDDLInvalidExpressionError(message + errors)

    def check_state_invariants(self, silent: bool=False) -> bool:
        '''Throws an exception if the state invariants are not satisfied.'''
        for (i, invariant) in enumerate(self.invariants):
            loc = self.invariant_names[i]
            sample, self.key, error, self.model_params = invariant(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if not self._to_bool(sample):
                if not silent:
                    raise RDDLStateInvariantNotSatisfiedError(
                        f'{loc} is not satisfied.')
                return False
        return True

    def check_action_preconditions(self, actions: Args, silent: bool=False) -> bool:
        '''Throws an exception if the action preconditions are not satisfied.'''
        subs = self.subs
        subs.update(actions) # its from compiler that super from initializer class that bulitd from pyRDDLGym.core.compiler.model
        
        for (i, precond) in enumerate(self.preconds):
            loc = self.precond_names[i]
            sample, self.key, error, self.model_params = precond(
                subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if not self._to_bool(sample): 
                if not silent:
                    raise RDDLActionPreconditionNotSatisfiedError(
                        f'{loc} is not satisfied for actions {actions}.')
                return False
        return True

    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for (i, terminal) in enumerate(self.terminals):
            loc = self.terminal_names[i]
            sample, self.key, error, self.model_params = terminal(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if self._to_bool(sample): 
                return True
        return False

    def sample_reward(self) -> float:
        reward, self.key, error, self.model_params = self.reward(
            self.subs, self.model_params, self.key)
        self.handle_error_code(error, 'reward function')
        return float(self._to_scalar(reward))

    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        rddl = self.rddl
        keep_tensors = self.keep_tensors
        subs = self.subs
        subs.update(actions)

        # compute CPFs in topological order
        for (cpf, expr, _) in self.cpfs:
            subs[cpf], self.key, error, self.model_params = expr(
                subs, self.model_params, self.key)
            self.handle_error_code(error, f'CPF <{cpf}>')

        # sample reward
        reward = self.sample_reward()

        # update state
        self.state = {}
        for (state, next_state) in rddl.next_state.items():
            # set the state = state' for the next epoch
            subs[state] = subs[next_state]


            # convert object integer to string representation 
            state_values = subs[state]
            view_values = state_values #why its save the value 
            if self.objects_as_strings:
                ptype = rddl.variable_ranges[state]
                if ptype not in RDDLValueInitializer.NUMPY_TYPES:
                    view_values = rddl.index_to_object_string_array(
                        ptype, self._to_numpy(state_values)) ### why do to it?
            # optional grounding of state dictionary
            if keep_tensors:
                self.state[state] = view_values
            else:
                tensorless = self._to_numpy(view_values) # again why to do it?
                self.state.update(rddl.ground_var_with_values(state, tensorless))
        # update observation
        if self._pomdp:
            obs = {}
            for var in rddl.observ_fluents:

                # convert object integer to string representation
                obs_values = subs[var]
                view_values = obs_values # again why its save the value 
                if self.objects_as_strings:
                    ptype = rddl.variable_ranges[var]
                    if ptype not in RDDLValueInitializer.NUMPY_TYPES:
                        view_values = rddl.index_to_object_string_array(ptype, self._to_numpy(obs_values))
                # optional grounding of observ-fluent dictionary   
                if keep_tensors:
                    obs[var] = view_values
                else:
                    obs.update(rddl.ground_var_with_values(var, self._to_numpy(view_values)))
        else:
            obs = self.state

        done = self.check_terminal_states()
        return obs, reward, done



    def _to_numpy(self, value: Any):
        if isinstance(value, torch.Tensor):
            data = value.detach()
            if data.device.type != 'cpu':
                data = data.cpu()
            return data.numpy()
        return value

    def _to_scalar(self, value: Any):
        if isinstance(value, torch.Tensor):
            return self._to_numpy(value).item()
        if isinstance(value, np.ndarray):
            return value.item()
        return value

    def _to_bool(self, value: Any) -> bool:
        return bool(self._to_scalar(value))
