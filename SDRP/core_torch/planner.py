"""Torch-native planner modeled after JaxBackpropPlanner (simplified).

This implements a minimal straight-line plan (SLP) optimizer using torch.autograd.
It relies on the TorchRDDLCompiler to build differentiable CPF/reward callables
and uses a torch optimizer (default Adam) to improve an open-loop action sequence.
"""

from __future__ import annotations

import math
import time
from abc import ABCMeta, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import torch
from torch import nn

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.policy import BaseAgent

from .compiler import TorchRDDLCompiler, TorchRDDLCompilerWithGrad
from .logic import FuzzyLogic

Bounds = Dict[str, Tuple[torch.Tensor, torch.Tensor]]
Kwargs = Dict[str, Any]
Params = Dict[str, nn.Parameter]


# ---------------------------------------------------------------------------
# Plan definitions
# ---------------------------------------------------------------------------


class TorchPlan(metaclass=ABCMeta):
    """Base class for torch policies/plans."""

    def __init__(self) -> None:
        self.train_policy: Optional[Callable] = None
        self.test_policy: Optional[Callable] = None
        self.projection: Optional[Callable] = None
        self.initializer: Optional[Callable] = None

    @abstractmethod
    def compile(self, compiled: TorchRDDLCompilerWithGrad,
                _bounds: Bounds,
                horizon: int) -> None:
        """Prepare policy functions given a compiled RDDL model."""
        raise NotImplementedError


class TorchStraightLinePlan(TorchPlan):
    """Open-loop sequence of actions parameterized directly."""

    def __init__(self,
                 initializer: Callable[[torch.Generator, Tuple[int, ...], torch.dtype], torch.Tensor]=None,
                 wrap_sigmoid: bool=True,
                 wrap_non_bool: bool=True) -> None:
        super().__init__()
        self._initializer = initializer
        self._wrap_sigmoid = wrap_sigmoid
        self._wrap_non_bool = wrap_non_bool
        self.bounds: Bounds = {}

    def compile(self, compiled: TorchRDDLCompilerWithGrad,
                _bounds: Bounds,
                horizon: int) -> None:
        rddl = compiled.rddl
        ranges = rddl.variable_ranges
        self.bounds = _bounds
        real_dtype = compiled.REAL

        def _init_params(key: torch.Generator, hyperparams: Dict[str, Any], subs):
            params: Params = {}
            for action in rddl.action_fluents:
                shape = compiled.init_values[action].shape
                full_shape = (horizon,) + tuple(shape)
                tensor = torch.empty(full_shape, dtype=real_dtype)
                if self._initializer is not None:
                    tensor = self._initializer(key, full_shape, real_dtype)
                else:
                    torch.nn.init.normal_(tensor, mean=0.0, std=0.01, generator=key)
                params[action] = nn.Parameter(tensor, requires_grad=True)
            return params

        def _train_policy(key, params: Params, hyperparams, step: int, subs):
            actions = {}
            for (var, tensor) in params.items():
                action = tensor[step]
                if ranges[var] == 'bool':
                    action = torch.sigmoid(action) if self._wrap_sigmoid else action
                else:
                    lower, upper = self.bounds.get(var, (None, None))
                    if lower is not None and upper is not None and self._wrap_non_bool:
                        action = lower + (upper - lower) * torch.sigmoid(action)
                actions[var] = action
            return actions

        def _test_policy(key, params: Params, hyperparams, step: int, subs):
            actions = {}
            for (var, tensor) in params.items():
                action = tensor[step]
                if ranges[var] == 'bool':
                    action = (torch.sigmoid(action) > 0.5).to(dtype=compiled.INT)
                else:
                    lower, upper = self.bounds.get(var, (None, None))
                    if lower is not None and upper is not None:
                        action = torch.clamp(action, lower, upper)
                actions[var] = action
            return actions

        def _projection(params: Params, hyperparams):
            for (var, tensor) in params.items():
                if ranges[var] == 'bool':
                    continue
                lower, upper = self.bounds.get(var, (None, None))
                if lower is not None and upper is not None and not self._wrap_non_bool:
                    tensor.data.copy_(torch.clamp(tensor.data, lower, upper))
            return params, True

        self.initializer = _init_params
        self.train_policy = _train_policy
        self.test_policy = _test_policy
        self.projection = _projection


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class TorchPlannerStatus(Enum):
    NORMAL = 0
    NO_PROGRESS = 1
    INVALID_GRADIENT = 2
    TIME_BUDGET_REACHED = 3
    ITER_BUDGET_REACHED = 4

    def is_terminal(self) -> bool:
        return self.value >= 2


class TorchBackpropPlanner:
    """Gradient-based planner using torch for a TorchPlan."""

    def __init__(self, rddl: RDDLLiftedModel,
                 plan: TorchPlan= None,
                 batch_size_train: int=32,
                 batch_size_test: Optional[int]=None,
                 rollout_horizon: Optional[int]=None,
                 optimizer: Callable[..., torch.optim.Optimizer]=torch.optim.Adam,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 clip_grad: Optional[float]=None,
                 logic: Optional[object]=None,
                 use64bit: bool=False,
                 logger=None) -> None:
        if plan is None:
            plan = TorchStraightLinePlan()
        if batch_size_test is None:
            batch_size_test = batch_size_train
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-3}
        self.rddl = rddl
        self.plan = plan
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.horizon = rollout_horizon
        self.optimizer_factory = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.clip_grad = clip_grad
        self.logic = logic if logic is not None else FuzzyLogic()
        self.use64bit = use64bit
        self.logger = logger
        self.real_dtype = torch.float64 if use64bit else torch.float32
        self.int_dtype = torch.int64 if use64bit else torch.int32

        self._compile_model()
        self._compile_optimizer()

    # ------------------------------------------------------------------
    # compilation
    # ------------------------------------------------------------------

    def _compile_model(self):
        self.compiled = TorchRDDLCompilerWithGrad(
            rddl=self.rddl,
            logic=self.logic,
            use64bit=self.use64bit
        )
        self.compiled.compile(log_expr=False, heading='RELAXED MODEL')

        self.test_compiled = TorchRDDLCompiler(
            rddl=self.rddl,
            logic=self.logic,
            use64bit=self.use64bit
        )
        self.test_compiled.compile(log_expr=False, heading='EXACT MODEL')

    def _compute_action_bounds(self, compiled: TorchRDDLCompiler) -> Bounds:
        bounds: Bounds = {}
        for (name, rng) in compiled.rddl.action_ranges.items():
            lower = upper = None
            if isinstance(rng, (tuple, list)):
                if len(rng) >= 2:
                    lower, upper = rng[0], rng[1]
            elif hasattr(rng, "__len__"):
                # numpy/torch arrays: treat min/max as bounds
                try:
                    lower = float(torch.as_tensor(rng).min().item())
                    upper = float(torch.as_tensor(rng).max().item())
                except Exception:
                    pass
            if lower is not None and upper is not None:
                lb = torch.as_tensor(lower, dtype=compiled.REAL)
                ub = torch.as_tensor(upper, dtype=compiled.REAL)
                bounds[name] = (lb, ub)
        return bounds

    def _batched_init_subs(self, compiled: TorchRDDLCompiler, batch_size: int):
        init = {}
        for (name, value) in compiled.init_values.items():
            tensor = torch.as_tensor(value, dtype=compiled.REAL)
            view = tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.dim()))
            init[name] = view
        for (state, next_state) in compiled.rddl.next_state.items():
            init[next_state] = init[state]
        return init

    def _compile_optimizer(self):
        action_bounds = self._compute_action_bounds(self.compiled)
        self.plan.compile(self.compiled, _bounds=action_bounds, horizon=self.horizon)
        self.train_policy = self.plan.train_policy
        self.test_policy = self.plan.test_policy

        self.train_subs_template = self._batched_init_subs(
            self.compiled, self.batch_size_train)
        self.test_subs_template = self._batched_init_subs(
            self.test_compiled, self.batch_size_test)

        self.train_rollouts = self._build_rollouts(
            compiled=self.compiled,
            policy_fn=self.train_policy,
            base_subs=self.train_subs_template
        )
        self.test_rollouts = self._build_rollouts(
            compiled=self.test_compiled,
            policy_fn=self.test_policy,
            base_subs=self.test_subs_template
        )

        self.train_loss = self._loss(self.train_rollouts)
        self.test_loss = self._loss(self.test_rollouts)
        self.update = self._update(self.train_loss)

    # ------------------------------------------------------------------
    # rollouts and loss
    # ------------------------------------------------------------------

    def _build_rollouts(self, compiled: TorchRDDLCompiler,
                        policy_fn: Callable,
                        base_subs: Dict[str, torch.Tensor]):
        rddl = compiled.rddl
        level_order = []
        for level in sorted(compiled.levels.keys()):
            level_order.extend(compiled.levels[level])

        def _rollouts(key: torch.Generator,
                      policy_params: Params,
                      policy_hyperparams: Dict[str, Any],
                      subs=None,
                      model_params: Optional[Dict[str, Any]]=None):
            if subs is None:
                subs = {k: v.clone() for (k, v) in base_subs.items()}
            if model_params is None:
                model_params = compiled.model_params
            local_subs = {k: v.clone() for (k, v) in subs.items()}
            rewards = []
            for step in range(self.horizon):
                actions = policy_fn(key, policy_params, policy_hyperparams, step, local_subs)
                local_subs.update(actions)
                for cpf in level_order:
                    fn = compiled.cpfs[cpf]
                    value, key, _, model_params = fn(local_subs, model_params, key)
                    local_subs[cpf] = value
                reward, key, _, model_params = compiled.reward(local_subs, model_params, key)
                rewards.append(reward)
                for (state, next_state) in rddl.next_state.items():
                    local_subs[state] = local_subs[next_state]
            reward_tensor = torch.stack(
                [self._ensure_tensor(r, compiled.REAL) for r in rewards], dim=1)
            return {'reward': reward_tensor, 'subs': local_subs}, model_params

        return _rollouts

    @staticmethod
    def _ensure_tensor(value: Any, dtype: torch.dtype):
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=dtype)

    def _loss(self, rollouts):
        def _plan_loss(key, policy_params, policy_hyperparams, subs, model_params):
            log, model_params = rollouts(
                key, policy_params, policy_hyperparams, subs, model_params)
            rewards = log['reward']
            returns = torch.sum(rewards, dim=1)
            loss_val = -returns.mean()
            return loss_val, (log, model_params)
        return _plan_loss

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------

    def _init_policy(self, key: torch.Generator,
                     policy_hyperparams: Optional[Dict[str, Any]]=None):
        if policy_hyperparams is None:
            policy_hyperparams = {action: torch.tensor(1.0, dtype=self.real_dtype)
                                  for action in self.rddl.action_fluents}
        params = self.plan.initializer(key, policy_hyperparams, self.train_subs_template)
        optimizer = self.optimizer_factory(params.values(), **self.optimizer_kwargs)
        return params, optimizer, policy_hyperparams

    def _update(self, loss_fn):
        projection = self.plan.projection

        def _step(key, params, policy_hyperparams, subs, model_params, optimizer):
            optimizer.zero_grad()
            loss_val, (log, model_params) = loss_fn(
                key, params, policy_hyperparams, subs, model_params)
            if not torch.isfinite(loss_val):
                zero_grads = {name: True for name in params}
                return params, optimizer, float('nan'), zero_grads, log, model_params
            loss_val.backward()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(params.values(), self.clip_grad)
            optimizer.step()
            if projection is not None:
                params, _ = projection(params, policy_hyperparams)
            zero_grads = {}
            for (name, tensor) in params.items():
                grad = tensor.grad
                zero_grads[name] = (grad is None or torch.allclose(
                    grad, torch.zeros_like(grad), atol=1e-8))
            return params, optimizer, float(loss_val.detach().cpu().item()), zero_grads, log, model_params

        return _step

    def optimize(self, *args, **kwargs):
        iterator = self.optimize_generator(*args, **kwargs)
        last = deque(iterator, maxlen=1)
        if last:
            return last.pop()
        return None

    def optimize_generator(self,
                           key: Optional[torch.Generator]=None,
                           epochs: int=100,
                           train_seconds: float=120.,
                           policy_hyperparams: Optional[Dict[str, Any]]=None,
                           print_progress: bool=True) -> Generator[Dict[str, Any], None, None]:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
            print_progress = False

        rng = key if key is not None else torch.Generator()
        rng.manual_seed(round(time.time() * 1000))
        params, optimizer, policy_hyperparams = self._init_policy(rng, policy_hyperparams)
        subs = self.train_subs_template
        model_params = self.compiled.model_params

        start_time = time.time()
        progress_bar = None
        if print_progress and tqdm is not None:
            progress_bar = tqdm(total=epochs, bar_format='{l_bar}{bar}| {elapsed} {postfix}')

        for it in range(epochs):
            status = TorchPlannerStatus.NORMAL
            params, optimizer, loss_val, zero_grads, log, model_params = self.update(
                rng, params, policy_hyperparams, subs, model_params, optimizer)
            if not math.isfinite(loss_val):
                status = TorchPlannerStatus.INVALID_GRADIENT
            elif all(zero_grads.values()):
                status = TorchPlannerStatus.NO_PROGRESS
            elapsed = time.time() - start_time
            if elapsed >= train_seconds:
                status = TorchPlannerStatus.TIME_BUDGET_REACHED
            if it >= epochs - 1:
                status = TorchPlannerStatus.ITER_BUDGET_REACHED

            callback = {
                'status': status,
                'iteration': it,
                'train_return': -loss_val,
                'params': params,
                'policy_hyperparams': policy_hyperparams,
                'key': rng,
                'log': log
            }

            if print_progress and progress_bar is not None:
                progress_bar.set_description(
                    f'{it:6} it / {-loss_val:12.6f} return / {status.name}')
                progress_bar.update(1)

            yield callback
            if status.is_terminal():
                break

        if print_progress and progress_bar is not None:
            progress_bar.close()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class TorchOfflineController(BaseAgent):
    """Wraps a TorchBackpropPlanner for training and evaluation."""

    def __init__(self, planner: TorchBackpropPlanner,
                 params: Optional[Params]=None,
                 policy_hyperparams: Optional[Dict[str, Any]]=None,
                 epochs: int=200,
                 train_seconds: float=120.,
                 print_summary: bool=False) -> None:
        super().__init__()
        self.planner = planner
        self.params = params
        self.policy_hyperparams = policy_hyperparams
        self.epochs = epochs
        self.train_seconds = train_seconds
        self.print_summary = print_summary
        self._eval_step = 0

    def train(self):
        if self.params is not None:
            return
        callback = self.planner.optimize(
            epochs=self.epochs,
            train_seconds=self.train_seconds,
            policy_hyperparams=self.policy_hyperparams,
            print_progress=self.print_summary
        )
        if callback is not None:
            self.params = callback['params']
            self.policy_hyperparams = callback['policy_hyperparams']

    def reset(self):
        """Reset internal step counter before a new episode."""
        self._eval_step = 0

    def sample_action(self, state):
        """BaseAgent interface: sample action given current state."""
        action = self.sample_action_eval(state, self._eval_step)
        self._eval_step += 1
        return action

    def sample_action_eval(self, state: Dict[str, Any], step: int):
        if self.params is None:
            self.train()
        subs = {k: torch.as_tensor(v) for (k, v) in state.items()}
        actions = self.planner.test_policy(
            torch.Generator(), self.params, self.policy_hyperparams, step, subs)
        return {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for (k, v) in actions.items()}

    def evaluate(self, env, episodes: int=1, seed: Optional[int]=None,
                 render: bool=False, verbose: bool=True):
        self.train()
        rewards = []
        for ep in range(episodes):
            if seed is not None:
                env.seed(seed + ep)
            state, *_ = env.reset()
            self.reset()
            total = 0.0
            for step in range(env.horizon):
                action = self.sample_action(state)
                state, reward, done, *_ = env.step(action)
                total += reward
                if render and hasattr(env, 'render'):
                    env.render()
                if done:
                    break
            rewards.append(total)
            if verbose:
                print(f'[TorchPlanner] episode {ep}: reward={total:.3f}')
        return {
            'mean': float(sum(rewards) / max(1, len(rewards))),
            'std': float(torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0),
            'rewards': rewards
        }
