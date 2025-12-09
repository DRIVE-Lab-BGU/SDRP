"""Torch-friendly compiler that wraps the default pyRDDLGym simulator."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.simulator import RDDLSimulatorPrecompiled


Args = Dict[str, Any]


class TorchRDDLCompiler:
    """Mimics the API of JaxRDDLCompiler while delegating evaluation to numpy."""

    ERROR_CODES = {'NORMAL': 0}

    def __init__(self, rddl: RDDLLiftedModel,
                 logger: Optional[Logger]=None,
                 python_functions: Optional[Dict[str, Callable]]=None,
                 use64bit: bool=False,
                 **_) -> None:
        self.rddl = rddl
        self.logger = logger
        self.python_functions = python_functions or {}
        self.use64bit = use64bit
        self.device = torch.device('cpu')
        self.real_dtype = torch.float64 if use64bit else torch.float32
        self.int_dtype = torch.int64
        self.INT = torch.int64 if use64bit else torch.int32
        self.REAL = torch.float64 if use64bit else torch.float32
        self.TORCH_TYPES = {
            'int': self.INT,
            'real': self.REAL,
            'bool': torch.bool
        }

        initializer = RDDLValueInitializer(rddl, logger=logger)
        self.init_values_numpy = initializer.initialize()
        self.init_values = self._tensorize_structure(self.init_values_numpy)

        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state=True, logger=logger)
        self.levels = sorter.compute_levels()
        tracer = RDDLObjectsTracer(rddl, logger=logger, cpf_levels=self.levels)
        self.traced = tracer.trace()

        self.numpy_simulator = RDDLSimulatorPrecompiled(
            rddl=rddl,
            init_values=self.init_values_numpy,
            levels=self.levels,
            trace_info=self.traced,
            python_functions=self.python_functions
        )

        self.model_params: Dict[str, Any] = {}
        self.invariants: List[Callable] = []
        self.preconditions: List[Callable] = []
        self.terminations: List[Callable] = []
        self.cpfs: Dict[str, Callable] = {}
        self.reward: Optional[Callable] = None

    def compile(self, log_expr: bool=False, log_jax_expr: bool=False,
                heading: str='SIMULATION MODEL') -> None:
        self.invariants = self._compile_constraints(self.rddl.invariants)
        self.preconditions = self._compile_constraints(self.rddl.preconditions)
        self.terminations = self._compile_constraints(self.rddl.terminations)
        self.cpfs = self._compile_cpfs()
        self.reward = self._compile_reward()

    def _compile_constraints(self, constraints) -> List[Callable]:
        return [self._wrap_expression(expr) for expr in constraints]

    def _compile_cpfs(self) -> Dict[str, Callable]:
        compiled = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                compiled[cpf] = self._wrap_expression(expr)
        return compiled

    def _compile_reward(self) -> Callable:
        return self._wrap_expression(self.rddl.reward)

    def compile_transition(self, *_, **__):
        def _torch_wrapped_transition(key, actions, subs, model_params):
            numpy_subs = self._to_numpy_structure(subs)
            numpy_actions = self._to_numpy_structure(actions)
            numpy_subs.update(numpy_actions)

            for (cpf, expr, _) in self.numpy_simulator.cpfs:
                numpy_subs[cpf] = self.numpy_simulator._sample(expr, numpy_subs)

            reward_value = self.numpy_simulator._sample(self.rddl.reward, numpy_subs)
            reward = self._to_tensor(reward_value)
            torch_subs = self._tensorize_structure(numpy_subs)
            log = {
                'reward': reward,
                'error': self.ERROR_CODES['NORMAL'],
                'fluents': {var: torch_subs[var] for var in self.rddl.state_fluents},
                'precondition': True,
                'invariant': True,
                'termination': False
            }
            return torch_subs, log, model_params

        return _torch_wrapped_transition

    def _wrap_expression(self, expr):
        def evaluator(subs: Args, params: Args, key: Optional[torch.Generator]):
            numpy_subs = self._to_numpy_structure(subs)
            value = self.numpy_simulator._sample(expr, numpy_subs)
            tensor = self._to_tensor(value)
            return tensor, key, self.ERROR_CODES['NORMAL'], params
        return evaluator

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------

    def _tensorize_structure(self, data: Any):
        if isinstance(data, dict):
            return {k: self._tensorize_structure(v) for (k, v) in data.items()}
        if isinstance(data, list):
            return [self._tensorize_structure(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._tensorize_structure(v) for v in data)
        return self._to_tensor(data)

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, np.ndarray):
            if value.dtype == np.object_:
                return torch.tensor(0.0, dtype=self.real_dtype, device=self.device)
            dtype = torch.bool if value.dtype == np.bool_ else (
                self.int_dtype if np.issubdtype(value.dtype, np.integer) else self.real_dtype)
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        if isinstance(value, (bool, np.bool_)):
            return torch.tensor(bool(value), dtype=torch.bool, device=self.device)
        if isinstance(value, (int, np.integer)):
            return torch.tensor(int(value), dtype=self.int_dtype, device=self.device)
        if isinstance(value, (float, np.floating)):
            dtype = torch.float64 if isinstance(value, np.float64) or self.use64bit else torch.float32
            return torch.tensor(float(value), dtype=dtype, device=self.device)
        return torch.tensor(value, dtype=self.real_dtype, device=self.device)

    def _to_numpy_structure(self, data: Any):
        if isinstance(data, dict):
            return {k: self._to_numpy_structure(v) for (k, v) in data.items()}
        if isinstance(data, list):
            return [self._to_numpy_structure(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._to_numpy_structure(v) for v in data)
        if isinstance(data, torch.Tensor):
            array = data.detach()
            if array.device.type != 'cpu':
                array = array.cpu()
            return array.numpy()
        return data


class TorchRDDLCompilerWithGrad(TorchRDDLCompiler):
    """Placeholder for compatibility with the Torch model learner."""

    def compile_transition(self, *args, **kwargs):
        return super().compile_transition(*args, **kwargs)
