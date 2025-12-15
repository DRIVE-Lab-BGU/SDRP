"""Torch port of pyRDDLGym_jax.core.model with a torch-native learner."""

import math
import time
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from pyRDDLGym.core.compiler.model import RDDLLiftedModel

from . import logic as torch_logic

try:
    from .compiler import TorchRDDLCompilerWithGrad  # type: ignore
except Exception:  # pragma: no cover
    TorchRDDLCompilerWithGrad = None  # type: ignore

Kwargs = Dict[str, Any]
State = Dict[str, torch.Tensor]
Action = Dict[str, torch.Tensor]
DataStream = Iterable[Tuple[State, Action, State]]
Params = Dict[str, torch.Tensor]
Callback = Dict[str, Any]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mean_squared_error() -> LossFunction:
    """Return a squared-error loss closure `loss(target, pred)`.

    Returns:
        LossFunction: Callable that expects `(target, prediction)` tensors and
            outputs `(target - prediction)^2`.

    Example:
        >>> loss = mean_squared_error()
        >>> loss(torch.tensor([1.0]), torch.tensor([0.5]))
        tensor([0.2500])
    """
    def _torch_mse(target, pred):
        return torch.square(target - pred)
    return _torch_mse


def binary_cross_entropy(eps: float=1e-6) -> LossFunction:
    """Return a numerically safe BCE closure `loss(target, pred)`.

    Args:
        eps (float): Small constant used to clamp predictions away from 0/1.

    Returns:
        LossFunction: Callable mapping `(target, prediction)` to BCE loss.

    Example:
        >>> loss = binary_cross_entropy()
        >>> loss(torch.tensor([1.0]), torch.tensor([0.8]))
        tensor([0.2231])
    """
    def _torch_bce(target, pred):
        pred = torch.clamp(pred, eps, 1.0 - eps)
        log_pred = torch.log(pred)
        log_not_pred = torch.log(1.0 - pred)
        return -target * log_pred - (1.0 - target) * log_not_pred
    return _torch_bce


def optax_loss(loss_fn: Callable[..., torch.Tensor], **kwargs) -> LossFunction:
    """Wrap an Optax-style loss to match the learner signature.

    Args:
        loss_fn: Callable taking predictions first and targets second.
        **kwargs: Extra keyword arguments forwarded to `loss_fn`.

    Returns:
        LossFunction: Callable taking `(target, pred)` and forwarding to Optax.

    Example:
        >>> def l2(pred, target):
        ...     return (pred - target) ** 2
        >>> loss = optax_loss(l2)
        >>> loss(torch.tensor([1.0]), torch.tensor([0.0]))
        tensor([1.])
    """
    def _wrapped_loss(target, pred):
        return loss_fn(pred, target, **kwargs)
    return _wrapped_loss


class TorchLearnerStatus(Enum):
    """Mirror of the Jax learner status codes."""
    NORMAL = 0
    NO_PROGRESS = 1
    INVALID_GRADIENT = 2
    TIME_BUDGET_REACHED = 3
    ITER_BUDGET_REACHED = 4

    def is_terminal(self) -> bool:
        """Whether the optimization loop should stop for this status.

        Returns:
            bool: True once the learner reached an unrecoverable issue.

        Example:
            >>> TorchLearnerStatus.NO_PROGRESS.is_terminal()
            False
        """
        return self.value >= 2


class TorchModelLearner:
    """Gradient-based learner for non-fluent parameters using torch."""

    def __init__(self, rddl: RDDLLiftedModel,
                 param_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
                 batch_size_train: int=32,
                 samples_per_datapoint: int=1,
                 optimizer: Callable[..., Optimizer]=torch.optim.Adam,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 initializer: Optional[Callable[[torch.Generator, Tuple[int, ...], torch.dtype], torch.Tensor]]=None,
                 wrap_non_bool: bool=True,
                 use64bit: bool=False,
                 bool_fluent_loss: LossFunction=binary_cross_entropy(),
                 real_fluent_loss: LossFunction=mean_squared_error(),
                 int_fluent_loss: LossFunction=mean_squared_error(),
                 logic: torch_logic.Logic=torch_logic.FuzzyLogic(),
                 model_params_reduction: Callable[[Any], Any]=lambda x: x[0],
                 compiler_factory: Optional[Callable[..., Any]]=None,
                 device: Optional[torch.device]=None) -> None:
        """Create a learner that fits non-fluent ranges with gradient descent.

        Args:
            rddl: Lifted RDDL description to mimic.
            param_ranges: Mapping of non-fluent name to `(lower, upper)` range.
            batch_size_train: Number of trajectories consumed per step.
            samples_per_datapoint: How many stochastic rollouts per datapoint.
            optimizer: Optimizer constructor such as `torch.optim.Adam`.
            optimizer_kwargs: Extra keyword args forwarded to the optimizer.
            initializer: Callable creating the initial tensor for each param.
            wrap_non_bool: If True, wrap params to enforce bounds analytically.
            use64bit: If True, prefer float64 tensors for precision.
            bool_fluent_loss: Loss used for boolean state prediction error.
            real_fluent_loss: Loss used for real-valued states.
            int_fluent_loss: Loss used for discrete integer states.
            logic: Logic backend injected into the compiler.
            model_params_reduction: Aggregator for per-sample hyperparameters.
            compiler_factory: Factory returning a Torch compiler with gradients.
            device: Torch device for parameters and computations.

        Example:
            >>> learner = TorchModelLearner(rddl_model, {'nf': (0.0, 1.0)})
            >>> callback = learner.optimize(data_stream)
            >>> callback['status']
            TorchLearnerStatus.NORMAL
        """
        self.rddl = rddl
        self.param_ranges = param_ranges.copy()
        self.batch_size_train = batch_size_train
        self.samples_per_datapoint = max(1, samples_per_datapoint)
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-3}
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_factory = optimizer
        self.initializer = initializer or self._default_initializer
        self.wrap_non_bool = wrap_non_bool
        self.use64bit = use64bit
        self.real_dtype = torch.float64 if use64bit else torch.float32
        self.bool_fluent_loss = bool_fluent_loss
        self.real_fluent_loss = real_fluent_loss
        self.int_fluent_loss = int_fluent_loss
        self.logic = logic
        self.model_params_reduction = model_params_reduction
        self.compiler_factory = compiler_factory or TorchRDDLCompilerWithGrad
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(round(time.time() * 1000))

        self._validate_param_ranges()
        self.step_fn = self._torch_compile_rddl()
        self.map_fn = self._torch_map()
        self.loss_fn = self._torch_loss(self.map_fn, self.step_fn)
        self.update_fn, self.project_fn = self._torch_update(self.loss_fn)
        self.init_fn, self.init_opt_fn = self._torch_init(self.project_fn)

    # ------------------------------------------------------------------
    # compilation helpers
    # ------------------------------------------------------------------

    def _default_initializer(self, key: torch.Generator,
                             shape: Tuple[int, ...],
                             dtype: torch.dtype) -> torch.Tensor:
        """Create a small Gaussian tensor for parameter initialization.

        Args:
            key: RNG used so each parameter receives a deterministic sample.
            shape: Desired parameter shape.
            dtype: Target dtype for the tensor.

        Returns:
            torch.Tensor: Random normal tensor matching the provided metadata.

        Example:
            >>> learner._default_initializer(torch.Generator(), (2,), torch.float32)
            tensor([ 0.0012, -0.0078])
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        return torch.nn.init.normal_(tensor, mean=0.0, std=0.01, generator=key)

    def _validate_param_ranges(self):
        """Ensure user supplied bounds are valid for every declared non-fluent.

        Raises:
            ValueError: If the name is unknown or bounds are malformed.

        Example:
            >>> learner._validate_param_ranges()  # no exception means ok
        """
        for (name, values) in self.param_ranges.items():
            if name not in self.rddl.non_fluents:
                raise ValueError(f'param_ranges key <{name}> is not a valid non-fluent.')
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError(f'param_ranges values with key <{name}> must be length-2 tuple.')
            lower, upper = values
            if lower is not None and upper is not None:
                lower_val = float(lower)
                upper_val = float(upper)
                if lower_val > upper_val:
                    raise ValueError(
                        f'param_ranges values with key <{name}> do not satisfy lower <= upper.')

    def _torch_compile_rddl(self):
        """Compile the lifted RDDL to differentiable PyTorch callables.

        Returns:
            Callable: Step function accepting `(key, params, subs, actions, hyperparams)`
            and returning stacked next-state predictions and hyperparameters.

        Example:
            >>> step_fn = learner._torch_compile_rddl()
            >>> next_state, hyper = step_fn(key, params, subs, acts, hyper)
        """
        if self.compiler_factory is None:
            raise ImportError('TorchRDDLCompilerWithGrad is required for TorchModelLearner.')
        self.compiled = self.compiler_factory(
            rddl=self.rddl,
            logic=self.logic,
            use64bit=self.use64bit,
            compile_non_fluent_exact=False,
            print_warnings=True
        )
        self.compiled.compile(log_expr=True, heading='RELAXED MODEL')
        step_fn = self._build_transition_function()

        def _torch_wrapped_step(key, param_fluents, subs, actions, hyperparams):
            local_subs = {name: (value.clone() if isinstance(value, torch.Tensor) else value)
                          for (name, value) in subs.items()}
            for (name, param) in param_fluents.items():
                local_subs[name] = param
            next_subs, _, hyperparams = step_fn(key, actions, local_subs, hyperparams)
            return next_subs, hyperparams

        def _torch_wrapped_batched_step(key, param_fluents, subs, actions, hyperparams):
            batch_outputs = []
            hyperparams_list = []
            batch_size = self._infer_batch_dim(subs)
            subkeys = self._split_generator(key, batch_size)
            for idx, subkey in enumerate(subkeys):
                sample_subs = self._index_structure(subs, idx)
                sample_actions = self._index_structure(actions, idx)
                next_subs, next_hyperparams = _torch_wrapped_step(
                    subkey, param_fluents, sample_subs, sample_actions, hyperparams)
                batch_outputs.append(next_subs)
                hyperparams_list.append(next_hyperparams)
            stacked_subs = self._stack_structure(batch_outputs)
            reduced_hyperparams = self._reduce_tree(hyperparams_list)
            return stacked_subs, reduced_hyperparams

        def _torch_wrapped_parallel_step(key, param_fluents, subs, actions, hyperparams):
            sample_outputs = []
            hyperparams_list = []
            subkeys = self._split_generator(key, self.samples_per_datapoint)
            for subkey in subkeys:
                next_subs, next_hyperparams = _torch_wrapped_batched_step(
                    subkey, param_fluents, subs, actions, hyperparams)
                sample_outputs.append(next_subs)
                hyperparams_list.append(next_hyperparams)
            stacked_samples = self._stack_structure(sample_outputs)
            reduced_hyperparams = self._reduce_tree(hyperparams_list)
            return stacked_samples, reduced_hyperparams

        return _torch_wrapped_parallel_step

    def _build_transition_function(self):
        """Create a deterministic transition that mirrors CPF evaluation order.

        Returns:
            Callable: Function `(key, actions, subs, hyperparams)` -> `(next_subs, log, hyperparams)`.

        Example:
            >>> transition = learner._build_transition_function()
            >>> next_subs, log, hyper = transition(key, actions, subs, {})
        """
        compiled = self.compiled
        level_keys = sorted(compiled.levels.keys())

        def _transition(key, actions, subs, hyperparams):
            local_subs = {name: (value.clone() if isinstance(value, torch.Tensor) else value)
                          for (name, value) in subs.items()}
            for (name, value) in actions.items():
                local_subs[name] = value

            for level in level_keys:
                for cpf in sorted(compiled.levels[level]):
                    fn = compiled.cpfs[cpf]
                    value, key, _, hyperparams = fn(local_subs, hyperparams, key)
                    local_subs[cpf] = value

            for (state, next_state) in self.rddl.next_state.items():
                local_subs[state] = local_subs[next_state]

            log = {'reward': torch.tensor(0.0, dtype=compiled.REAL, device=self.device)}
            return local_subs, log, hyperparams

        return _transition

    def _torch_map(self):
        """Map trainable `nn.Parameter` tensors to bounded non-fluent values.

        Returns:
            Callable: Function `params -> param_fluents` suitable for the model.

        Example:
            >>> map_fn = learner._torch_map()
            >>> param_fluents = map_fn({'nf': nn.Parameter(torch.zeros(1))})
        """
        case_indices = {}
        processed_ranges = {}
        if self.wrap_non_bool:
            for (name, (lower, upper)) in self.param_ranges.items():
                lval = -math.inf if lower is None else float(lower)
                uval = math.inf if upper is None else float(upper)
                processed_ranges[name] = (lval, uval)
                if math.isfinite(lval) and math.isfinite(uval):
                    case_indices[name] = 0
                elif math.isfinite(lval) and not math.isfinite(uval):
                    case_indices[name] = 1
                elif not math.isfinite(lval) and math.isfinite(uval):
                    case_indices[name] = 2
                else:
                    case_indices[name] = 3
            self.param_ranges = processed_ranges

        def _torch_params_to_fluents(params: Dict[str, nn.Parameter]):
            param_fluents = {}
            for (name, param) in params.items():
                value = param
                if self.rddl.variable_ranges[name] == 'bool':
                    param_fluents[name] = torch.sigmoid(value)
                else:
                    if self.wrap_non_bool:
                        lower, upper = self.param_ranges[name]
                        weight = torch.as_tensor(
                            lower, dtype=self.real_dtype, device=self.device)
                        span = torch.as_tensor(
                            upper - lower, dtype=self.real_dtype, device=self.device)
                        case = case_indices[name]
                        if case == 0:
                            param_fluents[name] = weight + span * torch.sigmoid(value)
                        elif case == 1:
                            param_fluents[name] = weight + (torch.nn.functional.elu(value) + 1.0)
                        elif case == 2:
                            ub = torch.as_tensor(upper, dtype=self.real_dtype, device=self.device)
                            param_fluents[name] = ub - (torch.nn.functional.elu(-value) + 1.0)
                        else:
                            param_fluents[name] = value
                    else:
                        lower, upper = self.param_ranges[name]
                        low = torch.as_tensor(
                            -math.inf if lower is None else lower,
                            dtype=self.real_dtype, device=self.device)
                        high = torch.as_tensor(
                            math.inf if upper is None else upper,
                            dtype=self.real_dtype, device=self.device)
                        param_fluents[name] = torch.clamp(value, low, high)
            return param_fluents

        return _torch_params_to_fluents

    def _torch_loss(self, map_fn, step_fn):
        """Build the differentiable loss that compares predicted and target states.

        Args:
            map_fn: Callable returning bounded parameters.
            step_fn: Callable returning next state predictions.

        Returns:
            Callable: Function `(key, params, subs, actions, next_fluents, hyperparams)`
            -> `(loss, hyperparams)`.

        Example:
            >>> loss_fn = learner._torch_loss(learner.map_fn, learner.step_fn)
            >>> loss, _ = loss_fn(key, params, subs, actions, targets, {})
        """
        def _torch_wrapped_batched_model_loss(key, param_fluents, subs, actions,
                                              next_fluents, hyperparams):
            next_subs, hyperparams = step_fn(
                key, param_fluents, subs, actions, hyperparams)
            total_loss = torch.zeros(1, dtype=self.real_dtype, device=self.device)
            count = max(1, len(next_fluents))
            for (name, next_value) in next_fluents.items():
                preds = next_subs[name].to(self.real_dtype)
                targets = self._ensure_tensor(next_value, dtype=self.real_dtype).unsqueeze(0)
                if self.rddl.variable_ranges[name] == 'bool':
                    loss_values = self.bool_fluent_loss(targets, preds)
                elif self.rddl.variable_ranges[name] == 'real':
                    loss_values = self.real_fluent_loss(targets, preds)
                else:
                    loss_values = self.int_fluent_loss(targets, preds)
                total_loss = total_loss + torch.mean(loss_values) / count
            return total_loss.squeeze(), hyperparams

        def _torch_wrapped_batched_loss(key, params, subs, actions,
                                        next_fluents, hyperparams):
            param_fluents = map_fn(params)
            loss, hyperparams = _torch_wrapped_batched_model_loss(
                key, param_fluents, subs, actions, next_fluents, hyperparams)
            return loss, hyperparams

        return _torch_wrapped_batched_loss

    def _torch_init(self, project_fn):
        """Create parameter initialization utilities for random or user guesses.

        Args:
            project_fn: Callable that enforces parameter constraints.

        Returns:
            Tuple[Callable, Callable]: `(init_fn, init_opt_fn)` where the first
            samples fresh parameters and the second wraps user guesses.

        Example:
            >>> init_params, init_opt = learner._torch_init(lambda x: x)
            >>> params, opt = init_params(torch.Generator())
        """
        def _init_params_optimizer(key: torch.Generator):
            params = {}
            names = list(self.param_ranges.keys())
            subkeys = self._split_generator(key, len(names))
            for (name, subkey) in zip(names, subkeys):
                init_value = self.initializer(
                    subkey, tuple(self._value_shape(self.compiled.init_values[name])),
                    self.real_dtype)
                params[name] = nn.Parameter(init_value.clone().requires_grad_(True))
            project_fn(params)
            optimizer = self._create_optimizer(params)
            return params, optimizer

        def _init_optimizer(params: Dict[str, torch.Tensor]):
            param_dict = {
                name: nn.Parameter(self._ensure_tensor(value, dtype=self.real_dtype).clone().requires_grad_(True))
                for (name, value) in params.items()
            }
            project_fn(param_dict)
            optimizer = self._create_optimizer(param_dict)
            return param_dict, optimizer

        return _init_params_optimizer, _init_optimizer

    def _torch_update(self, loss_fn):
        """Create an SGD update rule along with projection of constrained params.

        Args:
            loss_fn: Callable producing the loss for a mini-batch.

        Returns:
            Tuple[Callable, Callable]: `(update_fn, project_fn)` used in training.

        Example:
            >>> update_fn, project_fn = learner._torch_update(learner.loss_fn)
            >>> params, opt, loss, zeros, hyper = update_fn(key, params, subs, acts, tgt, {}, opt)
        """
        def _project_params(params: Dict[str, nn.Parameter]):
            if self.wrap_non_bool:
                return params
            for (name, tensor) in params.items():
                if self.rddl.variable_ranges[name] == 'bool':
                    continue
                lower, upper = self.param_ranges[name]
                low = -math.inf if lower is None else float(lower)
                high = math.inf if upper is None else float(upper)
                tensor.data.copy_(torch.clamp(
                    tensor.data,
                    min=torch.as_tensor(low, dtype=self.real_dtype, device=self.device),
                    max=torch.as_tensor(high, dtype=self.real_dtype, device=self.device)
                ))
            return params

        def _torch_wrapped_params_update(key, params, subs, actions,
                                         next_fluents, hyperparams, optimizer):
            optimizer.zero_grad()
            loss, hyperparams = loss_fn(
                key, params, subs, actions, next_fluents, hyperparams)
            if not torch.isfinite(loss):
                zero_grads = {name: True for name in params}
                return params, optimizer, float('nan'), zero_grads, hyperparams
            loss.backward()
            zero_grads = {}
            for (name, tensor) in params.items():
                grad = tensor.grad
                zero_grads[name] = (grad is None or torch.allclose(
                    grad, torch.zeros_like(grad), atol=1e-8))
            optimizer.step()
            _project_params(params)
            return params, optimizer, float(loss.detach().cpu().item()), zero_grads, hyperparams

        return _torch_wrapped_params_update, _project_params

    def _batched_init_subs(self):
        """Replicate initial state fluents across the training batch dimension.

        Returns:
            Dict[str, torch.Tensor]: Batched initial substitutions keyed by fluent.

        Example:
            >>> subs = learner._batched_init_subs()
            >>> subs['state_fluent'].shape
            torch.Size([learner.batch_size_train, ...])
        """
        init_train = {}
        for (name, value) in self.compiled.init_values.items():
            tensor = self._ensure_tensor(value, dtype=self.real_dtype)
            repeat_factors = (self.batch_size_train,) + (1,) * tensor.dim()
            expanded = tensor.unsqueeze(0).repeat(*repeat_factors)
            init_train[name] = expanded
        for (state, next_state) in self.rddl.next_state.items():
            init_train[next_state] = init_train[state]
        return init_train

    # ------------------------------------------------------------------
    # optimization API
    # ------------------------------------------------------------------

    def optimize(self, *args, **kwargs) -> Optional[Callback]:
        """Run `optimize_generator` to completion and return the final callback.

        Returns:
            Optional[Callback]: Last yielded dictionary or `None` when empty.

        Example:
            >>> result = learner.optimize(data_stream, epochs=10)
            >>> result['train_loss']
            0.1234
        """
        iterator = self.optimize_generator(*args, **kwargs)
        last = deque(iterator, maxlen=1)
        if last:
            return last.pop()
        return None

    def optimize_generator(self, data: DataStream,
                           key: Optional[torch.Generator]=None,
                           epochs: int=999999,
                           train_seconds: float=120.,
                           guess: Optional[Params]=None,
                           print_progress: bool=True) -> Generator[Callback, None, None]:
        """Stream optimization callbacks so callers can monitor training live.

        Args:
            data: Iterable of `(state, action, next_state)` batches.
            key: Optional RNG overriding the learner seed.
            epochs: Maximum number of iterations to process.
            train_seconds: Wall-clock budget for optimization.
            guess: Optional initial parameter values.
            print_progress: Whether to render a tqdm progress bar.

        Yields:
            Callback: Dictionary with status, iteration, parameters, etc.

        Example:
            >>> for callback in learner.optimize_generator(data, epochs=5):
            ...     print(callback['iteration'], callback['train_loss'])
        """
        try:
            from tqdm import tqdm
        except Exception:  # pragma: no cover
            tqdm = None
            print_progress = False

        start_time = time.time()
        elapsed_outside_loop = 0.0
        rng = key if key is not None else self.generator
        subs = self._batched_init_subs()
        if guess is None:
            subkey = self._split_once(rng)
            params, opt_state = self.init_fn(subkey)
        else:
            params, opt_state = self.init_opt_fn(guess)
        hyperparams = self.compiled.model_params
        progress_bar = None
        if print_progress and tqdm is not None:
            progress_bar = tqdm(total=100, bar_format='{l_bar}{bar}| {elapsed} {postfix}')

        for (it, (states, actions, next_states)) in enumerate(data):
            status = TorchLearnerStatus.NORMAL
            subs.update(self._tensorize_structure(states))
            actions_tensor = self._tensorize_structure(actions)
            next_states_tensor = self._tensorize_structure(next_states)

            subkey = self._split_once(rng)
            params, opt_state, loss, zero_grads, hyperparams = self.update_fn(
                subkey, params, subs, actions_tensor, next_states_tensor, hyperparams, opt_state)

            if not math.isfinite(loss):
                status = TorchLearnerStatus.INVALID_GRADIENT
            if any(zero_grads.values()):
                status = TorchLearnerStatus.NO_PROGRESS
            elapsed = time.time() - start_time - elapsed_outside_loop
            if elapsed >= train_seconds:
                status = TorchLearnerStatus.TIME_BUDGET_REACHED
            if it >= epochs - 1:
                status = TorchLearnerStatus.ITER_BUDGET_REACHED

            progress_percent = 100 * min(
                1.0, max(0.0, elapsed / max(train_seconds, 1e-6), it / max(epochs - 1, 1)))
            callback = {
                'status': status,
                'iteration': it,
                'train_loss': loss,
                'params': params,
                'param_fluents': self.map_fn(params),
                'key': rng,
                'progress': progress_percent
            }
            if print_progress and progress_bar is not None:
                progress_bar.set_description(
                    f'{it:7} it / {loss:12.8f} train / {status.value} status', refresh=False)
                progress_bar.set_postfix_str(
                    f'{(it + 1) / (elapsed + 1e-6):.2f}it/s', refresh=False)
                progress_bar.update(progress_percent - progress_bar.n)

            start_out = time.time()
            yield callback
            elapsed_outside_loop += (time.time() - start_out)

            if status.is_terminal():
                break

    def evaluate_loss(self, data: DataStream,
                      key: Optional[torch.Generator],
                      param_fluents: Params) -> float:
        """Compute the average loss of learned parameters on provided data.

        Args:
            data: Iterable of `(state, action, next_state)` tuples.
            key: Optional RNG to make rollouts deterministic.
            param_fluents: Dictionary of mapped parameter tensors.

        Returns:
            float: Mean loss across the dataset.

        Example:
            >>> loss = learner.evaluate_loss(data, None, learner.map_fn(params))
        """
        rng = key if key is not None else self.generator
        subs = self._batched_init_subs()
        hyperparams = self.compiled.model_params
        mean_loss = 0.0
        for (it, (states, actions, next_states)) in enumerate(data):
            subs.update(self._tensorize_structure(states))
            actions_tensor = self._tensorize_structure(actions)
            next_states_tensor = self._tensorize_structure(next_states)
            subkey = self._split_once(rng)
            preds, hyperparams = self.step_fn(
                subkey, param_fluents, subs, actions_tensor, hyperparams)
            loss = torch.zeros(1, dtype=self.real_dtype, device=self.device)
            count = max(1, len(next_states_tensor))
            for (name, target) in next_states_tensor.items():
                preds_tensor = preds[name].to(self.real_dtype)
                target_tensor = target.to(self.real_dtype).unsqueeze(0)
                if self.rddl.variable_ranges[name] == 'bool':
                    loss_values = self.bool_fluent_loss(target_tensor, preds_tensor)
                elif self.rddl.variable_ranges[name] == 'real':
                    loss_values = self.real_fluent_loss(target_tensor, preds_tensor)
                else:
                    loss_values = self.int_fluent_loss(target_tensor, preds_tensor)
                loss = loss + torch.mean(loss_values) / count
            loss_value = float(loss.detach().cpu().item())
            mean_loss += (loss_value - mean_loss) / (it + 1)
        return mean_loss

    def learned_model(self, param_fluents: Params) -> RDDLLiftedModel:
        """Produce a Python RDDL model with non-fluents replaced by learned values.

        Args:
            param_fluents: Dictionary mapping non-fluent names to tensors.

        Returns:
            RDDLLiftedModel: Deep copy of the original with updated non-fluents.

        Example:
            >>> fitted = learner.learned_model(learner.map_fn(params))
            >>> fitted.non_fluents['capacity']
            3.2
        """
        model = deepcopy(self.rddl)
        for (name, values) in param_fluents.items():
            tensor = torch.as_tensor(values)
            if model.variable_ranges[name] == 'bool':
                updated = (tensor > 0.5).to(torch.bool)
            elif model.variable_ranges[name] == 'int':
                updated = torch.round(tensor).to(torch.int64)
            else:
                updated = tensor.to(self.real_dtype)
            flattened = updated.reshape(-1).tolist()
            if not model.variable_params[name]:
                assert len(flattened) == 1
                flattened = flattened[0]
            model.non_fluents[name] = flattened
        return model

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def seed(self, seed: int) -> None:
        """Set the internal RNG seed so later training runs are reproducible.

        Args:
            seed (int): New seed value forwarded to the PyTorch generator.

        Example:
            >>> learner.seed(42)
        """
        self.generator.manual_seed(seed)

    def _ensure_tensor(self, value: Any, dtype: Optional[torch.dtype]=None) -> torch.Tensor:
        """Convert arbitrary data to a tensor on the learner device and dtype.

        Args:
            value: Python/NumPy/Torch input.
            dtype: Optional dtype override.

        Returns:
            torch.Tensor: Tensor suitable for downstream computations.

        Example:
            >>> learner._ensure_tensor([1, 2]).device
            device(type='cpu')
        """
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        tensor = torch.as_tensor(value, dtype=dtype or self.real_dtype, device=self.device)
        return tensor

    def _tensorize_structure(self, structure: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert a `{name: value}` mapping into tensors using `_ensure_tensor`.

        Args:
            structure: Dictionary keyed by fluent name.

        Returns:
            Dict[str, torch.Tensor]: Tensorized structure.

        Example:
            >>> learner._tensorize_structure({'f': 1.0})['f']
            tensor(1.)
        """
        return {name: self._ensure_tensor(value, dtype=self._var_dtype(name))
                for (name, value) in structure.items()}

    def _split_generator(self, base: torch.Generator, num: int):
        """Create `num` new generators by splitting the provided base RNG.

        Args:
            base: Generator to draw fresh seeds from.
            num: Number of sub-generators requested.

        Returns:
            List[torch.Generator]: Deterministic generators.

        Example:
            >>> gens = learner._split_generator(torch.Generator(), 2)
            >>> len(gens)
            2
        """
        seeds = torch.randint(
            0, 2**31 - 1, (num,), generator=base).tolist()
        result = []
        for seed in seeds:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(seed))
            result.append(gen)
        return result

    def _split_once(self, base: torch.Generator):
        """Convenience helper returning a single generator from `_split_generator`.

        Args:
            base: Generator to split.

        Returns:
            torch.Generator: Newly seeded generator.

        Example:
            >>> learner._split_once(torch.Generator())
        """
        return self._split_generator(base, 1)[0]

    def _value_shape(self, value: Any):
        """Return the tensor shape corresponding to the provided sample value.

        Args:
            value: Python/NumPy/Torch structure.

        Returns:
            Tuple[int, ...]: Shape tuple useful for parameter initialization.

        Example:
            >>> learner._value_shape(torch.zeros(2))
            torch.Size([2])
        """
        tensor = torch.as_tensor(value)
        return tensor.shape

    def _create_optimizer(self, params: Dict[str, nn.Parameter]):
        """Instantiate the user-provided optimizer for the parameter list.

        Args:
            params: Dictionary of learnable tensors.

        Returns:
            Optimizer: Torch optimizer ready to step gradients.

        Example:
            >>> learner._create_optimizer({'nf': nn.Parameter(torch.zeros(1))})
        """
        parameter_list = list(params.values())
        return self.optimizer_factory(parameter_list, **self.optimizer_kwargs)

    def _index_structure(self, structure: Dict[str, torch.Tensor], index: int):
        """Take the `index`-th sample from every tensor in a batched structure.

        Args:
            structure: Dictionary where tensors share their leading batch dim.
            index: Integer batch index to slice.

        Returns:
            Dict[str, torch.Tensor]: Structure representing a single sample.

        Example:
            >>> learner._index_structure({'x': torch.arange(6).view(3, 2)}, 1)
            {'x': tensor([2, 3])}
        """
        result = {}
        for (name, value) in structure.items():
            tensor = self._ensure_tensor(value)
            if tensor.dim() == 0:
                result[name] = tensor
            else:
                result[name] = tensor[index]
        return result

    def _stack_structure(self, samples: Any):
        """Stack a list of samples (possibly nested dicts) along a new dimension.

        Args:
            samples: List of tensors or dictionaries of tensors.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: Batched structure.

        Example:
            >>> learner._stack_structure([{'x': torch.tensor(1)}, {'x': torch.tensor(2)}])
            {'x': tensor([1, 2])}
        """
        if not samples:
            return {}
        if isinstance(samples[0], dict):
            keys = samples[0].keys()
            return {k: self._stack_structure([sample[k] for sample in samples]) for k in keys}
        tensors = [self._ensure_tensor(sample) for sample in samples]
        return torch.stack(tensors, dim=0)

    def _reduce_tree(self, values: Any):
        """Reduce nested hyperparameter outputs using `model_params_reduction`.

        Args:
            values: List of dictionaries or tensors.

        Returns:
            Any: Reduced structure, commonly a mean or first element.

        Example:
            >>> learner._reduce_tree([{'reward': 1.}, {'reward': 2.}])
            {'reward': 1.5}
        """
        if not values:
            return {}
        if isinstance(values[0], dict):
            return {k: self._reduce_tree([value[k] for value in values]) for k in values[0]}
        return self.model_params_reduction(values)

    def _var_dtype(self, name: str) -> torch.dtype:
        """Return the dtype to use for the fluent `name` (currently `real_dtype`).

        Args:
            name: Fluent identifier (unused but retained for API parity).

        Returns:
            torch.dtype: Preferred dtype for tensors.

        Example:
            >>> learner._var_dtype('some_fluent')
            torch.float32
        """
        del name
        return self.real_dtype

    def _infer_batch_dim(self, structure: Dict[str, torch.Tensor]) -> int:
        """Infer the leading batch size from any tensor in the structure.

        Args:
            structure: Dictionary containing tensors of shape `(batch, ...)`.

        Returns:
            int: Observed batch dimension or fallback to `batch_size_train`.

        Example:
            >>> learner._infer_batch_dim({'x': torch.zeros(5, 3)})
            5
        """
        for value in structure.values():
            tensor = self._ensure_tensor(value)
            if tensor.dim() > 0:
                return int(tensor.shape[0])
        return self.batch_size_train
