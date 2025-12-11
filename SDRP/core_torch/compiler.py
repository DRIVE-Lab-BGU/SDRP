"""Torch-native RDDL compiler producing differentiable PyTorch callables.
mplemented a torch-native compiler (core_torch/compiler.py) that mirrors the Jax structure while producing eager,
 autograd-friendly PyTorch callables. TorchRDDLCompiler now:
Builds torch tensors for initial values, dependency levels, and tracer metadata; 
stores CPFs/reward/invariants/preconditions/terminations exactly like the Jax compiler
 so TorchRDDLSimulator can consume them.
Recursively compiles constants, pvars (including sliced/broadcast cases),
 arithmetic/relational/logical/aggregation/function/control nodes, python-function calls,
and selectable random variables (Uniform/Normal/Exponential/Bernoulli) 
into functions returning (value, key, error_code, params).
Supports optional fuzzy-logic backends by routing 
arithmetic/logical/control ops through user-provided hooks; falls back to torch ops otherwise.
Keeps everything in eager torch without .
numpy() conversions, so gradients can flow through the compiled CPFs and reward.
Limitations: transition compilation still raises NotImplementedError; 
switch nodes require a usable default if a case is None;
 external python functions are evaluated eagerly 
 but we don’t replicate the full captured-variable broadcasting from the Jax version.
 Next steps could include expanding the random-variable set,
 adding full pyfunc broadcasting support, and implementing planner-ready transition compilation if needed.
"""
####################################################################
# this file soports gradient computation using torch autograd ######
####################################################################



from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.debug.exception import (
    print_stack_trace,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger


Args = Dict[str, Any]
CallableExpr = Callable[[Args, Dict[str, Any], Optional[torch.Generator]],
                        Tuple[torch.Tensor, Optional[torch.Generator], int, Dict[str, Any]]]


class TorchRDDLCompiler:
    """Compiles RDDL expressions into eager PyTorch callables."""

    ERROR_CODES = {'NORMAL': 0}

    def __init__(self, rddl: RDDLLiftedModel,
                 logger: Optional[Logger]=None,
                 python_functions: Optional[Dict[str, Callable]]=None,
                 use64bit: bool=False,
                 fuzzy_logic: Optional[object]=None,
                 **_) -> None:
        self.rddl = rddl
        self.logger = logger
        self.python_functions = python_functions or {}
        self.fuzzy_logic = fuzzy_logic
        self.use64bit = use64bit

        self.INT = torch.int64 if use64bit else torch.int32
        self.REAL = torch.float64 if use64bit else torch.float32
        self.TORCH_TYPES = {
            'int': self.INT,
            'real': self.REAL,
            'bool': torch.bool
        }

        self.init_values: Dict[str, torch.Tensor] = {}
        self.cpfs: Dict[str, CallableExpr] = {}
        self.reward: Optional[CallableExpr] = None
        self.invariants: List[CallableExpr] = []
        self.preconditions: List[CallableExpr] = []
        self.terminations: List[CallableExpr] = []
        self.model_params: Dict[str, Any] = {}
        self.levels = None
        self.traced = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, log_expr: bool=False, log_jax_expr: bool=False,
                heading: str='') -> None:
        initializer = RDDLValueInitializer(self.rddl, logger=self.logger)
        init_values_np = initializer.initialize()
        self.init_values = self._tensorize_structure(init_values_np)

        sorter = RDDLLevelAnalysis(self.rddl, allow_synchronous_state=True,
                                   logger=self.logger)
        self.levels = sorter.compute_levels()
        tracer = RDDLObjectsTracer(self.rddl, logger=self.logger,
                                   cpf_levels=self.levels)
        self.traced = tracer.trace()
        init_params: Dict[str, Any] = {}
        self.model_params = init_params

        self.invariants = [self._torch(expr, init_params)
                           for expr in self.rddl.invariants]
        self.preconditions = [self._torch(expr, init_params)
                              for expr in self.rddl.preconditions]
        self.terminations = [self._torch(expr, init_params)
                             for expr in self.rddl.terminations]
        self.cpfs = self._compile_cpfs(init_params)
        self.reward = self._torch(self.rddl.reward, init_params)

    def compile_transition(self, *args, **kwargs):
        raise NotImplementedError(
            'Transition compilation for planning is not yet implemented.')

    # ------------------------------------------------------------------
    # Expression dispatch
    # ------------------------------------------------------------------

    def _torch(self, expr, init_params, dtype=None) -> CallableExpr:
        etype, _ = expr.etype
        if etype == 'constant':
            fn = self._torch_constant(expr)
        elif etype == 'pvar':
            fn = self._torch_pvar(expr, init_params)
        elif etype == 'arithmetic':
            fn = self._torch_arithmetic(expr, init_params)
        elif etype == 'relational':
            fn = self._torch_relational(expr, init_params)
        elif etype == 'boolean':
            fn = self._torch_logical(expr, init_params)
        elif etype == 'aggregation':
            fn = self._torch_aggregation(expr, init_params)
        elif etype == 'func':
            fn = self._torch_functional(expr, init_params)
        elif etype == 'pyfunc':
            fn = self._torch_pyfunc(expr, init_params)
        elif etype == 'control':
            fn = self._torch_control(expr, init_params)
        elif etype == 'randomvar':
            fn = self._torch_random(expr, init_params)
        else:
            raise RDDLNotImplementedError(
                f'Expression type {etype} is not supported.\n' + print_stack_trace(expr))

        if dtype is not None:
            def _cast(subs, params, key):
                value, key, err, params = fn(subs, params, key)
                return value.to(dtype), key, err, params
            return _cast
        return fn

    # ------------------------------------------------------------------
    # Leaves
    # ------------------------------------------------------------------

    def _torch_constant(self, expr) -> CallableExpr:
        cached_value = self.traced.cached_sim_info(expr)
        tensor = self._ensure_tensor(cached_value)

        def _fn(subs, params, key):
            return tensor, key, self.ERROR_CODES['NORMAL'], params

        return _fn

    def _torch_pvar(self, expr, init_params) -> CallableExpr:
        var, pvars = expr.args
        is_value, cached_info = self.traced.cached_sim_info(expr)

        if is_value:
            tensor = self._ensure_tensor(cached_info)

            def _fn(subs, params, key):
                return tensor, key, self.ERROR_CODES['NORMAL'], params

            return _fn

        if cached_info is None:
            def _scalar(subs, params, key):
                value = self._ensure_tensor(subs[var])
                return value, key, self.ERROR_CODES['NORMAL'], params

            return _scalar

        slices, axis, shape, op_code, op_args = cached_info
        tracer = RDDLObjectsTracer.NUMPY_OP_CODE

        if slices and op_code == tracer.NESTED_SLICE:
            compiled_slices = [
                self._torch(arg, init_params) if _slice is None
                else self._torch_slice(_slice) for (arg, _slice) in zip(pvars, slices)
            ]

            def _nested(subs, params, key):
                value = self._ensure_tensor(subs[var])
                new_slices = []
                error = self.ERROR_CODES['NORMAL']
                for inner in compiled_slices:
                    idx, key, err, params = inner(subs, params, key)
                    if isinstance(idx, torch.Tensor):
                        new_slices.append(idx.to(dtype=torch.long))
                    else:
                        new_slices.append(idx)
                    error |= err
                tuple_slices = tuple(new_slices)
                sample = value[tuple_slices]
                return sample, key, error, params

            return _nested

        def _non_nested(subs, params, key):
            sample = self._ensure_tensor(subs[var])
            if slices:
                sample = sample[slices]
            if axis:
                current = sample
                for ax in sorted(axis):
                    current = torch.unsqueeze(current, dim=ax)
                sample = current.expand(shape)
            if op_code == tracer.EINSUM:
                equation = op_args[0]
                operands = op_args[1:] if len(op_args) > 1 else ()
                sample = torch.einsum(equation, sample, *operands)
            elif op_code == tracer.TRANSPOSE:
                sample = sample.permute(op_args)
            return sample, key, self.ERROR_CODES['NORMAL'], params

        return _non_nested

    def _torch_slice(self, slice_value):
        if isinstance(slice_value, (slice, type(Ellipsis), type(None))):
            stored = slice_value
        else:
            stored = self._ensure_tensor(slice_value)

        def _fn(subs, params, key):
            return stored, key, self.ERROR_CODES['NORMAL'], params

        return _fn

    # ------------------------------------------------------------------
    # Helpers for operations
    # ------------------------------------------------------------------

    def _apply_unary(self, name: str, value: torch.Tensor):
        if self.fuzzy_logic is not None and hasattr(self.fuzzy_logic, name):
            return getattr(self.fuzzy_logic, name)(value)
        if name == 'neg':
            return -value
        if name == 'logical_not':
            return torch.logical_not(value)
        raise ValueError(f'Unsupported unary op {name}.')

    def _apply_binary(self, name: str, lhs: torch.Tensor, rhs: torch.Tensor):
        if self.fuzzy_logic is not None and hasattr(self.fuzzy_logic, name):
            return getattr(self.fuzzy_logic, name)(lhs, rhs)
        if name == 'add':
            return torch.add(lhs, rhs)
        if name == 'sub':
            return torch.sub(lhs, rhs)
        if name == 'mul':
            return torch.mul(lhs, rhs)
        if name == 'div':
            return torch.div(lhs, rhs)
        if name == 'pow':
            return torch.pow(lhs, rhs)
        if name == 'lt':
            return torch.lt(lhs, rhs)
        if name == 'le':
            return torch.le(lhs, rhs)
        if name == 'gt':
            return torch.gt(lhs, rhs)
        if name == 'ge':
            return torch.ge(lhs, rhs)
        if name == 'eq':
            return torch.eq(lhs, rhs)
        if name == 'ne':
            return torch.ne(lhs, rhs)
        if name == 'logical_and':
            return torch.logical_and(lhs, rhs)
        if name == 'logical_or':
            return torch.logical_or(lhs, rhs)
        raise ValueError(f'Unsupported binary op {name}.')

    def _apply_control_if(self, pred: torch.Tensor,
                          then_value: torch.Tensor,
                          else_value: torch.Tensor):
        if self.fuzzy_logic is not None and hasattr(self.fuzzy_logic, 'if_then_else'):
            return self.fuzzy_logic.if_then_else(pred, then_value, else_value)
        return torch.where(pred.to(dtype=self.REAL) > 0.5, then_value, else_value)

    def _apply_control_switch(self, pred: torch.Tensor, cases: torch.Tensor):
        if self.fuzzy_logic is not None and hasattr(self.fuzzy_logic, 'switch'):
            return self.fuzzy_logic.switch(pred, cases)
        pred_long = pred.to(dtype=torch.long).unsqueeze(0)
        reference = cases[:1]
        expanded_index = pred_long.expand_as(reference)
        gathered = torch.gather(cases, 0, expanded_index)
        return gathered.squeeze(0)

    def _aggregate(self, op: str, tensor: torch.Tensor, axes: Optional[Sequence[int]]):
        if axes is None:
            axes = tuple(range(tensor.dim()))
        axes = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
        result = tensor
        for axis in sorted(axes, reverse=True):
            if axis is None:
                continue
            if op == 'sum':
                result = torch.sum(result, dim=axis)
            elif op == 'forall':
                result = torch.all(result.bool(), dim=axis)
            elif op == 'exists':
                result = torch.any(result.bool(), dim=axis)
        return result

    # ------------------------------------------------------------------
    # Arithmetic / Logical / Relational
    # ------------------------------------------------------------------

    def _torch_arithmetic(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        args = [self._torch(arg, init_params) for arg in expr.args]

        if len(args) == 1 and op == '-':
            def _neg(subs, params, key):
                value, key, err, params = args[0](subs, params, key)
                return self._apply_unary('neg', value), key, err, params
            return _neg

        if len(args) < 2:
            raise RDDLNotImplementedError(
                f'Arithmetic operator {op} requires at least two arguments.\n' +
                print_stack_trace(expr))

        def _fn(subs, params, key):
            value, key, err, params = args[0](subs, params, key)
            for arg_fn in args[1:]:
                rhs, key, err_rhs, params = arg_fn(subs, params, key)
                err |= err_rhs
                if op == '+':
                    value = self._apply_binary('add', value, rhs)
                elif op == '-':
                    value = self._apply_binary('sub', value, rhs)
                elif op == '*':
                    value = self._apply_binary('mul', value, rhs)
                elif op == '/':
                    value = self._apply_binary('div', value, rhs)
                else:
                    raise RDDLNotImplementedError(
                        f'Arithmetic operator {op} is not supported.\n' +
                        print_stack_trace(expr))
            return value, key, err, params

        return _fn

    def _torch_relational(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        lhs, rhs = expr.args
        lhs_fn = self._torch(lhs, init_params)
        rhs_fn = self._torch(rhs, init_params)

        def _fn(subs, params, key):
            left, key, err1, params = lhs_fn(subs, params, key)
            right, key, err2, params = rhs_fn(subs, params, key)
            if op == '<':
                value = self._apply_binary('lt', left, right)
            elif op == '<=':
                value = self._apply_binary('le', left, right)
            elif op == '>':
                value = self._apply_binary('gt', left, right)
            elif op == '>=':
                value = self._apply_binary('ge', left, right)
            elif op == '==':
                value = self._apply_binary('eq', left, right)
            elif op == '~=':
                value = self._apply_binary('ne', left, right)
            else:
                raise RDDLNotImplementedError(
                    f'Relational operator {op} is not supported.\n' +
                    print_stack_trace(expr))
            return value, key, err1 | err2, params

        return _fn

    def _torch_logical(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        args = [self._torch(arg, init_params) for arg in expr.args]

        if len(args) == 1 and op == '~':
            def _not(subs, params, key):
                value, key, err, params = args[0](subs, params, key)
                return self._apply_unary('logical_not', value.bool()), key, err, params
            return _not

        if len(args) < 2:
            raise RDDLNotImplementedError(
                f'Logical operator {op} requires at least two arguments.\n' +
                print_stack_trace(expr))

        def _fn(subs, params, key):
            value, key, err, params = args[0](subs, params, key)
            value = value.bool()
            for arg_fn in args[1:]:
                rhs, key, err_rhs, params = arg_fn(subs, params, key)
                rhs = rhs.bool()
                err |= err_rhs
                if op in {'^', '&'}:
                    value = self._apply_binary('logical_and', value, rhs)
                elif op == '|':
                    value = self._apply_binary('logical_or', value, rhs)
                else:
                    raise RDDLNotImplementedError(
                        f'Logical operator {op} is not supported.\n' +
                        print_stack_trace(expr))
            return value, key, err, params

        return _fn

    # ------------------------------------------------------------------
    # Aggregation / Functional
    # ------------------------------------------------------------------

    def _torch_aggregation(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        *_, arg = expr.args
        _, axes = self.traced.cached_sim_info(expr)
        arg_fn = self._torch(arg, init_params)

        def _fn(subs, params, key):
            value, key, err, params = arg_fn(subs, params, key)
            if op == 'sum':
                reduced = self._aggregate('sum', value, axes)
            elif op == 'forall':
                reduced = self._aggregate('forall', value, axes)
            elif op == 'exists':
                reduced = self._aggregate('exists', value, axes)
            else:
                raise RDDLNotImplementedError(
                    f'Aggregation {op} not supported.\n' + print_stack_trace(expr))
            return reduced, key, err, params

        return _fn

    def _torch_functional(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        if len(expr.args) == 1:
            arg_fn = self._torch(expr.args[0], init_params)

            def _unary(subs, params, key):
                val, key, err, params = arg_fn(subs, params, key)
                value = self._apply_function_unary(op, val)
                return value, key, err, params

            return _unary

        elif len(expr.args) == 2:
            lhs_fn = self._torch(expr.args[0], init_params)
            rhs_fn = self._torch(expr.args[1], init_params)

            def _binary(subs, params, key):
                lhs, key, err1, params = lhs_fn(subs, params, key)
                rhs, key, err2, params = rhs_fn(subs, params, key)
                value = self._apply_function_binary(op, lhs, rhs)
                return value, key, err1 | err2, params

            return _binary

        raise RDDLNotImplementedError(
            f'Functional operator {op} is not supported.\n' + print_stack_trace(expr))

    def _apply_function_unary(self, op: str, value: torch.Tensor) -> torch.Tensor:
        if self.fuzzy_logic is not None and hasattr(self.fuzzy_logic, op):
            return getattr(self.fuzzy_logic, op)(value)
        funcs = {
            'abs': torch.abs,
            'exp': torch.exp,
            'ln': torch.log,
            'floor': torch.floor,
            'ceil': torch.ceil,
            'round': torch.round,
            'sqrt': torch.sqrt,
            'sin': torch.sin,
            'cos': torch.cos,
            'tan': torch.tan,
            'asin': torch.asin,
            'acos': torch.acos,
            'atan': torch.atan,
            'sinh': torch.sinh,
            'cosh': torch.cosh,
            'tanh': torch.tanh,
            'sgn': torch.sign
        }
        if op not in funcs:
            raise RDDLNotImplementedError(f'Unary function {op} is not supported.')
        return funcs[op](value)

    def _apply_function_binary(self, op: str, lhs: torch.Tensor,
                               rhs: torch.Tensor) -> torch.Tensor:
        if self.fuzzy_logic is not None and hasattr(self.fuzzy_logic, op):
            return getattr(self.fuzzy_logic, op)(lhs, rhs)
        funcs = {
            'min': torch.minimum,
            'max': torch.maximum,
            'pow': torch.pow,
            'div': torch.floor_divide,
            'mod': torch.remainder,
            'hypot': lambda a, b: torch.sqrt(a * a + b * b),
            'log': lambda a, b: torch.log(a) / torch.log(b)
        }
        if op not in funcs:
            raise RDDLNotImplementedError(f'Binary function {op} is not supported.')
        return funcs[op](lhs, rhs)

    # ------------------------------------------------------------------
    # External python functions
    # ------------------------------------------------------------------

    def _torch_pyfunc(self, expr, init_params) -> CallableExpr:
        _, pyfunc_name = expr.etype
        pyfunc = self.python_functions.get(pyfunc_name)
        if pyfunc is None:
            raise RDDLUndefinedVariableError(
                f'Undefined Python function <{pyfunc_name}>.\n' + print_stack_trace(expr))
        captured_vars, args = expr.args
        compiled_args = [self._torch(arg, init_params) for arg in args]

        def _fn(subs, params, key):
            values: List[torch.Tensor] = []
            error = self.ERROR_CODES['NORMAL']
            for arg in compiled_args:
                val, key, err, params = arg(subs, params, key)
                error |= err
                values.append(val)
            result = pyfunc(*values)
            tensor = self._ensure_tensor(result)
            return tensor, key, error, params

        return _fn

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _torch_control(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        if op == 'if':
            pred, then_expr, else_expr = expr.args
            pred_fn = self._torch(pred, init_params)
            then_fn = self._torch(then_expr, init_params)
            else_fn = self._torch(else_expr, init_params)

            def _fn(subs, params, key):
                pred_val, key, err_pred, params = pred_fn(subs, params, key)
                then_val, key, err_then, params = then_fn(subs, params, key)
                else_val, key, err_else, params = else_fn(subs, params, key)
                result = self._apply_control_if(pred_val, then_val, else_val)
                return result, key, err_pred | err_then | err_else, params

            return _fn

        elif op == 'switch':
            pred, *_ = expr.args
            cases, default = self.traced.cached_sim_info(expr)
            pred_fn = self._torch(pred, init_params)
            case_fns = [None if arg is None else self._torch(arg, init_params)
                        for arg in cases]
            default_fn = None if default is None else self._torch(default, init_params)

            def _fn(subs, params, key):
                pred_val, key, err, params = pred_fn(subs, params, key)
                evaluated_cases = []
                total_err = err
                default_value = None
                if default_fn is not None:
                    default_value, key, err_def, params = default_fn(subs, params, key)
                    total_err |= err_def
                for case_fn in case_fns:
                    if case_fn is None:
                        if default_value is None:
                            raise RDDLNotImplementedError(
                                'Switch case missing value and default is None.\n' +
                                print_stack_trace(expr))
                        evaluated_cases.append(default_value)
                    else:
                        val, key, err_case, params = case_fn(subs, params, key)
                        total_err |= err_case
                        evaluated_cases.append(val)
                stacked = torch.stack(evaluated_cases, dim=0)
                result = self._apply_control_switch(pred_val, stacked)
                return result, key, total_err, params

            return _fn

        raise RDDLNotImplementedError(
            f'Control operator {op} is not supported.\n' + print_stack_trace(expr))

    # ------------------------------------------------------------------
    # Random variables
    # ------------------------------------------------------------------

    def _torch_random(self, expr, init_params) -> CallableExpr:
        _, name = expr.etype
        args = [self._torch(arg, init_params) for arg in expr.args]

        def _fn(subs, params, key):
            values = []
            total_err = self.ERROR_CODES['NORMAL']
            for fn in args:
                val, key, err, params = fn(subs, params, key)
                values.append(val)
                total_err |= err
            generator = self._ensure_generator(key, values[0].device if values else torch.device('cpu'))
            if name == 'Uniform':
                low, high = torch.broadcast_tensors(values[0], values[1])
                rand = torch.rand(high.shape, generator=generator,
                                  device=high.device, dtype=self.REAL)
                sample = low + (high - low) * rand
            elif name == 'Normal':
                mean, var = torch.broadcast_tensors(values[0], values[1])
                std = torch.sqrt(torch.clamp(var, min=1e-8))
                eps = torch.randn(mean.shape, generator=generator,
                                  device=mean.device, dtype=self.REAL)
                sample = mean + std * eps
            elif name == 'Exponential':
                scale = values[0]
                rand = torch.rand(scale.shape, generator=generator,
                                  device=scale.device, dtype=self.REAL)
                sample = -scale * torch.log(torch.clamp(rand, min=1e-8))
            elif name == 'Bernoulli':
                probs = torch.clamp(values[0], 0.0, 1.0)
                rand = torch.rand(probs.shape, generator=generator,
                                  device=probs.device, dtype=self.REAL)
                sample = (rand <= probs).to(dtype=self.REAL)
            else:
                raise RDDLNotImplementedError(
                    f'Random variable {name} is not supported.\n' + print_stack_trace(expr))
            return sample, generator, total_err, params

        return _fn

    # ------------------------------------------------------------------
    # CPF / Reward compilation
    # ------------------------------------------------------------------

    def _compile_cpfs(self, init_params) -> Dict[str, CallableExpr]:
        cpfs: Dict[str, CallableExpr] = {}
        for cpfs_in_level in self.levels.values():
            for cpf in cpfs_in_level:
                _, expr = self.rddl.cpfs[cpf]
                cpfs[cpf] = self._torch(expr, init_params)
        return cpfs

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _ensure_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=self.REAL if value.dtype.is_floating_point else value.dtype)
        if isinstance(value, np.ndarray):
            if value.dtype == np.object_:
                raise TypeError('Object arrays are not supported in torch compilation.')
            if np.issubdtype(value.dtype, np.bool_):
                dtype = torch.bool
            elif np.issubdtype(value.dtype, np.integer):
                dtype = self.INT
            elif np.issubdtype(value.dtype, np.floating):
                dtype = self.REAL
            else:
                dtype = self.REAL
            return torch.as_tensor(value, dtype=dtype)
        if isinstance(value, (bool, np.bool_)):
            return torch.tensor(bool(value), dtype=torch.bool)
        if isinstance(value, (int, np.integer)):
            return torch.tensor(int(value), dtype=self.INT)
        if isinstance(value, (float, np.floating)):
            return torch.tensor(float(value), dtype=self.REAL)
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(value, dtype=self.REAL)
        return torch.tensor(value)

    def _tensorize_structure(self, data: Any):
        if isinstance(data, dict):
            return {k: self._tensorize_structure(v) for (k, v) in data.items()}
        if isinstance(data, list):
            return [self._tensorize_structure(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._tensorize_structure(v) for v in data)
        return self._ensure_tensor(data)

    def _ensure_generator(self, key: Optional[torch.Generator], device: torch.device):
        if key is not None:
            return key
        device_type = device.type if isinstance(device, torch.device) else 'cpu'
        generator = torch.Generator(device=device_type)
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item()
        generator.manual_seed(int(seed))
        return generator


class TorchRDDLCompilerWithGrad(TorchRDDLCompiler):
    """Gradient-aware compiler placeholder (shares implementation)."""

    pass
