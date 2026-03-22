from __future__ import annotations

import inspect
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.debug.logger import Logger

try:
    from .compiler import TorchRDDLCompiler
except ImportError:  # pragma: no cover - fallback for script-style execution
    from compiler import TorchRDDLCompiler

TensorDict = Dict[str, torch.Tensor]
PolicyOutput = Union[TensorDict, Tuple[TensorDict, Any]]
PolicyFn = Callable[..., PolicyOutput]


@dataclass
class RolloutTrace:
    """Container with the full trajectory produced by a rollout."""

    observations: List[TensorDict]
    actions: List[TensorDict]
    rewards: List[torch.Tensor]
    terminals: List[bool]
    final_observation: TensorDict
    final_subs: Dict[str, Any]
    policy_state: Any
    model_params: Dict[str, Any]

    @property
    def return_(self) -> torch.Tensor:
        if not self.rewards:
            return torch.tensor(0.0)
        return torch.stack(self.rewards, dim=0).sum(dim=0)


class TorchRolloutCell(nn.Module):
    """One transition of the RDDL dynamics.

    The simulator/compiler keeps the full environment state inside `subs`.
    In RNN terms:

    * hidden state  -> `subs`
    * observation   -> projection of `subs`
    * input         -> action
    * next hidden   -> updated `subs`
    """

    def __init__(self,
                 rddl_model: RDDLLiftedModel,
                 key: Optional[torch.Generator]=None,
                 logger: Optional[Logger]=None,
                 device: Optional[Union[str, torch.device]]=None,
                 **compiler_args) -> None:
        super().__init__()
        if key is None:
            key = torch.Generator()
            key.manual_seed(round(time.time() * 1000))
        self.key = key
        self.rddl = rddl_model
        self.logger = logger
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.compiler_args = compiler_args

        compiled = TorchRDDLCompiler(rddl_model, logger=logger, **compiler_args)
        compiled.compile(log_expr=False, heading='ROLLOUT MODEL')

        self.compiler = compiled
        self.step_fn = compiled.compile_transition(cache_path_info=False)
        self.init_values = self._clone_structure(compiled.init_values)
        self.model_params = self._clone_structure(compiled.model_params)
        self.observed_fluents = tuple(
            rddl_model.observ_fluents if rddl_model.observ_fluents else rddl_model.state_fluents
        )
        self.noop_actions = {
            name: self._clone_value(value)
            for (name, value) in self.init_values.items()
            if rddl_model.variable_types[name] == 'action-fluent'
        }

    def reset(self,
              initial_state: Optional[Dict[str, Any]]=None,
              initial_subs: Optional[Dict[str, Any]]=None,
              model_params: Optional[Dict[str, Any]]=None
              ) -> Tuple[Dict[str, Any], TensorDict, Dict[str, Any]]:
        """Create a fresh hidden state for a rollout.

        If you only know the initial observation/state, pass it in `initial_state`.
        The method overlays it on top of the compiler init values and also syncs the
        corresponding next-state fluents, so the first transition starts consistently.
        """
        if initial_state is not None and initial_subs is not None:
            raise ValueError('Pass either initial_state or initial_subs, not both.')

        if initial_subs is None:
            subs = self._clone_structure(self.init_values)
        else:
            subs = self._clone_structure(initial_subs)

        if initial_state is not None:
            for (name, value) in initial_state.items():
                if name not in self.rddl.state_fluents:
                    raise KeyError(f'<{name}> is not a valid state fluent.')
                if name not in subs:
                    raise KeyError(f'Hidden state does not contain state fluent <{name}>.')
                coerced = self._coerce_like(value, subs[name], name)
                subs[name] = coerced
                next_name = self.rddl.next_state.get(name)
                if next_name is not None and next_name in subs:
                    subs[next_name] = self._clone_value(coerced)

        if model_params is None:
            local_model_params = self._clone_structure(self.model_params)
        else:
            local_model_params = self._clone_structure(model_params)

        observation = self.observe(subs)
        return subs, observation, local_model_params

    def observe(self, subs: Dict[str, Any]) -> TensorDict:
        """Project the full hidden state to the observation exposed to the policy."""
        return {
            name: self._coerce_obs_value(subs[name], name)
            for name in self.observed_fluents
        }

    def step(self,
             subs: Dict[str, Any],
             actions: Optional[Dict[str, Any]]=None,
             model_params: Optional[Dict[str, Any]]=None
             ) -> Tuple[Dict[str, Any], TensorDict, torch.Tensor, bool, Dict[str, Any]]:
        """Run one transition starting from an explicit hidden state."""
        local_subs = self._clone_structure(subs)
        local_model_params = self._clone_structure(
            self.model_params if model_params is None else model_params
        )
        prepared_actions = self.prepare_actions(actions)

        next_subs, log, next_model_params = self.step_fn(
            self.key, prepared_actions, local_subs, local_model_params
        )
        reward = self._ensure_tensor(log['reward'])
        done = self._to_bool(log.get('termination', False))
        next_obs = self.observe(next_subs)
        return next_subs, next_obs, reward, done, next_model_params

    def prepare_actions(self, actions: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        if actions is None:
            return {name: self._clone_value(value) for (name, value) in self.noop_actions.items()}

        prepared = {name: self._clone_value(value) for (name, value) in self.noop_actions.items()}
        for (name, value) in actions.items():
            if name not in prepared:
                raise KeyError(
                    f'<{name}> is not a valid lifted action fluent. '
                    'Pass lifted action names, not grounded action names.'
                )
            prepared[name] = self._coerce_like(value, prepared[name], name)
        return prepared

    def _coerce_obs_value(self, value: Any, name: str) -> torch.Tensor:
        del name
        if isinstance(value, torch.Tensor):
            return value.clone()
        return self._ensure_tensor(value)

    def _coerce_like(self, value: Any, reference: Any, name: str) -> Any:
        if isinstance(reference, torch.Tensor):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value, device=reference.device)
            if tensor.shape != reference.shape:
                raise ValueError(
                    f'Value for <{name}> must have shape {tuple(reference.shape)}, '
                    f'got {tuple(tensor.shape)}.'
                )
            if reference.dtype == torch.bool:
                return tensor.bool()
            return tensor.to(dtype=reference.dtype, device=reference.device)
        return value

    @staticmethod
    def _ensure_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.all(value.bool()).item())
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)

    @staticmethod
    def _clone_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if hasattr(value, 'copy'):
            return value.copy()
        return deepcopy(value)

    @classmethod
    def _clone_structure(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: cls._clone_structure(v) for (k, v) in value.items()}
        if isinstance(value, list):
            return [cls._clone_structure(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_structure(v) for v in value)
        return cls._clone_value(value)


class TorchRollout(nn.Module):
    """Unroll the environment like an RNN over a fixed horizon."""

    def __init__(self,
                 rddl_model: RDDLLiftedModel,
                 horizon: Optional[int]=None,
                 key: Optional[torch.Generator]=None,
                 logger: Optional[Logger]=None,
                 device: Optional[Union[str, torch.device]]=None,
                 **compiler_args) -> None:
        super().__init__()
        self.rddl = rddl_model
        self.horizon = rddl_model.horizon if horizon is None else horizon
        self.cell = TorchRolloutCell(
            rddl_model=rddl_model,
            key=key,
            logger=logger,
            device=device,
            **compiler_args,
        )

    @property
    def noop_actions(self) -> Dict[str, Any]:
        return self.cell.noop_actions

    def reset(self,
              initial_state: Optional[Dict[str, Any]]=None,
              initial_subs: Optional[Dict[str, Any]]=None,
              model_params: Optional[Dict[str, Any]]=None
              ) -> Tuple[Dict[str, Any], TensorDict, Dict[str, Any]]:
        return self.cell.reset(
            initial_state=initial_state,
            initial_subs=initial_subs,
            model_params=model_params,
        )

    def step(self,
             subs: Dict[str, Any],
             actions: Optional[Dict[str, Any]]=None,
             model_params: Optional[Dict[str, Any]]=None
             ) -> Tuple[Dict[str, Any], TensorDict, torch.Tensor, bool, Dict[str, Any]]:
        return self.cell.step(subs=subs, actions=actions, model_params=model_params)

    def forward(self,
                policy: PolicyFn,
                initial_state: Optional[Dict[str, Any]]=None,
                initial_subs: Optional[Dict[str, Any]]=None,
                model_params: Optional[Dict[str, Any]]=None,
                policy_state: Any=None) -> RolloutTrace:
        subs, observation, local_model_params = self.reset(
            initial_state=initial_state,
            initial_subs=initial_subs,
            model_params=model_params,
        )

        observations: List[TensorDict] = []
        actions_log: List[TensorDict] = []
        rewards: List[torch.Tensor] = []
        terminals: List[bool] = []

        for step in range(self.horizon):
            observations.append(observation)
            raw_action, policy_state = self._call_policy(
                policy, observation, step, policy_state
            )
            actions = self.cell.prepare_actions(raw_action)
            subs, observation, reward, done, local_model_params = self.step(
                subs=subs,
                actions=actions,
                model_params=local_model_params,
            )
            actions_log.append(actions)
            rewards.append(reward)
            terminals.append(done)
            if done:
                break

        return RolloutTrace(
            observations=observations,
            actions=actions_log,
            rewards=rewards,
            terminals=terminals,
            final_observation=observation,
            final_subs=subs,
            policy_state=policy_state,
            model_params=local_model_params,
        )

    @staticmethod
    def _call_policy(policy: PolicyFn,
                     observation: TensorDict,
                     step: int,
                     policy_state: Any) -> Tuple[TensorDict, Any]:
        target = policy.forward if isinstance(policy, nn.Module) else policy
        signature = inspect.signature(target)
        params = list(signature.parameters.values())
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)
        positional = [
            param for param in params
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if has_varargs or len(positional) >= 3:
            output = policy(observation, step, policy_state)
        elif len(positional) == 2:
            output = policy(observation, step)
        else:
            output = policy(observation)

        if isinstance(output, tuple) and len(output) == 2:
            actions, next_policy_state = output
            return actions, next_policy_state
        return output, policy_state
