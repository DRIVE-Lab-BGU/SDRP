import time
from typing import Callable, Dict, Optional, Union, Any

import numpy as np
import torch

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.parser.expr import Value
from pyRDDLGym.core.simulator import RDDLSimulator

# Typed alias reused throughout the wrapper to highlight accepted value formats.
Args = Dict[str, Union[np.ndarray, torch.Tensor, Value, float, int, bool]]


class RDDLTorchSimulator(RDDLSimulator):
    '''Wrapper around the default pyRDDLGym simulator exposing a Torch-friendly API.'''

    def __init__(self, rddl: RDDLPlanningModel,
                 logger: Optional[Logger]=None,
                 keep_tensors: bool=False,
                 objects_as_strings: bool=True,
                 python_functions: Optional[Dict[str, Callable]]=None,
                 device: Optional[Union[str, torch.device]]=None,
                 tensor_dtype: torch.dtype=torch.float32) -> None:
        '''Creates a simulator that returns torch tensors and accepts torch actions.

        :param rddl: compiled RDDL planning model
        :param logger: optional logger passed to the numpy simulator
        :param keep_tensors: if True observations/states keep the vectorized shape
        :param objects_as_strings: whether enum values remain readable strings
        :param python_functions: external user-defined functions
        :param device: torch device for returned tensors (defaults to CPU)
        :param tensor_dtype: default dtype for floating point tensors
        '''
        # Initialize the underlying numpy-based simulator first.
        super(RDDLTorchSimulator, self).__init__(
            rddl=rddl,
            logger=logger,
            keep_tensors=keep_tensors,
            objects_as_strings=objects_as_strings,
            python_functions=python_functions)

        # Resolve the torch device where tensors should live (CPU default).
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.tensor_dtype = tensor_dtype

        # Torch RNG mirrors numpy RNG to give deterministic behavior when seeding.
        generator_device = (
            'cuda' if self.device.type == 'cuda' and torch.cuda.is_available()
            else 'cpu'
        )
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(round(time.time() * 1000))

    def seed(self, seed: int) -> None:
        '''Re-seed both numpy and torch random number generators.'''
        super(RDDLTorchSimulator, self).seed(seed)
        self.generator.manual_seed(seed)
        torch.manual_seed(seed)

    @property
    def states(self) -> Args:
        '''Returns the current state as torch tensors.'''
        base_state = super(RDDLTorchSimulator, self).states
        return self._tensorize_structure(base_state)

    def prepare_actions_for_sim(self, actions: Args) -> Args:
        '''Coerce user actions into numpy arrays before delegating to base sim.'''
        torch_free = {k: self._ensure_numpy(v) for (k, v) in actions.items()}
        return super(RDDLTorchSimulator, self).prepare_actions_for_sim(torch_free)

    def check_action_preconditions(self, actions: Args,
                                   silent: bool=False) -> bool:
        '''Run default precondition checks after stripping torch tensors.'''
        torch_free = {k: self._ensure_numpy(v) for (k, v) in actions.items()}
        return super(RDDLTorchSimulator, self).check_action_preconditions(
            torch_free, silent=silent)

    def reset(self):
        '''Reset simulator and convert the resulting numpy state to torch.'''
        obs, done = super(RDDLTorchSimulator, self).reset()
        return self._tensorize_structure(obs), done

    def step(self, actions: Args):
        '''Perform a simulator step accepting torch actions and returning torch state.'''
        numpy_actions = {k: self._ensure_numpy(v) for (k, v) in actions.items()}
        obs, reward, done = super(RDDLTorchSimulator, self).step(numpy_actions)
        return (
            self._tensorize_structure(obs),
            self._to_torch_scalar(reward),
            done
        )

    def sample_reward(self) -> torch.Tensor:
        reward = super(RDDLTorchSimulator, self).sample_reward()
        return self._to_torch_scalar(reward)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _ensure_numpy(self, value: Any):
        '''Detach torch tensors and recursively bring structures back to numpy.'''
        if isinstance(value, torch.Tensor):
            array = value.detach()
            if array.device.type != 'cpu':
                array = array.cpu()
            return array.numpy()
        if isinstance(value, dict):
            return {k: self._ensure_numpy(v) for (k, v) in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [self._ensure_numpy(v) for v in value]
            return type(value)(converted)
        return value

    def _tensorize_structure(self, data: Any):
        '''Mirror nested dict/list/tuple structures while casting leaves to tensors.'''
        if isinstance(data, dict):
            return {k: self._tensorize_structure(v) for (k, v) in data.items()}
        if isinstance(data, list):
            return [self._tensorize_structure(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._tensorize_structure(v) for v in data)
        return self._to_tensor(data)

    def _resolve_dtype_from_numpy(self, dtype: np.dtype) -> torch.dtype:
        '''Map numpy dtypes to best-effort torch dtype equivalents.'''
        if np.issubdtype(dtype, np.bool_):
            return torch.bool
        if np.issubdtype(dtype, np.integer):
            return torch.long
        if np.issubdtype(dtype, np.floating):
            return torch.float64 if dtype == np.float64 else self.tensor_dtype
        return self.tensor_dtype

    def _to_tensor(self, value: Any):
        '''Convert scalar or ndarray inputs to torch tensors on the configured device.'''
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, np.ndarray):
            if value.dtype == np.object_ or np.issubdtype(value.dtype, np.str_):
                return value.copy()
            dtype = self._resolve_dtype_from_numpy(value.dtype)
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        if isinstance(value, (np.bool_, bool)):
            return torch.tensor(bool(value), dtype=torch.bool, device=self.device)
        if isinstance(value, (np.integer, int)):
            return torch.tensor(int(value), dtype=torch.long, device=self.device)
        if isinstance(value, (np.floating, float)):
            dtype = torch.float64 if isinstance(value, np.float64) else self.tensor_dtype
            return torch.tensor(float(value), dtype=dtype, device=self.device)
        return value

    def _to_torch_scalar(self, value: Any) -> torch.Tensor:
        '''Cast rewards or scalar metadata to a torch tensor.'''
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, (np.bool_, bool)):
            return torch.tensor(bool(value), dtype=torch.bool, device=self.device)
        if isinstance(value, (np.integer, int)):
            return torch.tensor(int(value), dtype=torch.long, device=self.device)
        return torch.tensor(float(value), dtype=self.tensor_dtype, device=self.device)
