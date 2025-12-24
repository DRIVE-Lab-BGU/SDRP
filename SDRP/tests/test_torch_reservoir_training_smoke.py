import itertools
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from pyRDDLGym.core.compiler.model import RDDLLiftedModel  # noqa: E402
from pyRDDLGym.core.parser.parser import RDDLParser  # noqa: E402

import core_torch.model as torch_model  # noqa: E402

TorchModelLearner = torch_model.TorchModelLearner


class _TorchNativeLogic:
    """Placeholder logic so compiler falls back to raw torch ops."""
    pass


def _load_reservoir_rddl() -> str:
    """Load and sanitize the reservoir domain/instance pair for testing."""
    domain_path = os.path.join(ROOT, "instances", "reservoir", "domain.rddl")
    instance_path = os.path.join(ROOT, "instances", "reservoir", "instance_1.rddl")
    with open(domain_path, "r", encoding="utf-8", errors="replace") as f:
        domain = f.read()
    with open(instance_path, "r", encoding="utf-8", errors="replace") as f:
        instance = f.read()
    domain = domain.replace("-domain", "domain", 1)
    domain = domain.replace("\u00a0", " ")
    instance = instance.replace("\u00a0", " ")
    domain = domain.replace("rain(?r) =  abs[Normal(0, RAIN_VAR(?r))];", "rain(?r) = 0;")
    return f"{domain}\n{instance}"


def _build_lifted_model() -> RDDLLiftedModel:
    parser = RDDLParser(None, verbose=False)
    parser.build()
    rddl = parser.parse(_load_reservoir_rddl())
    return RDDLLiftedModel(rddl)


def _numpy_non_fluents(model: RDDLLiftedModel, names):
    return {name: np.asarray(model.non_fluents[name], dtype=np.float32) for name in names}


def _reservoir_data_iterator(batch_size: int, rddl: RDDLLiftedModel, learnable):
    """Yield synthetic data matching the reservoir shapes for a quick smoke test."""
    base = TorchModelLearner(
        rddl=rddl,
        param_ranges={name: (None, None) for name in learnable},
        batch_size_train=batch_size,
        samples_per_datapoint=1,
        optimizer_kwargs={"lr": 1e-3},
        wrap_non_bool=False,
        logic=_TorchNativeLogic(),
    )
    init_subs = base._batched_init_subs()
    state_shapes = {name: init_subs[name].shape for name in rddl.state_fluents}
    action_shapes = {name: init_subs[name].shape for name in rddl.action_fluents}
    rng = np.random.default_rng(round(time.time() * 1000))
    while True:
        states = {
            name: rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)
            for (name, shape) in state_shapes.items()
        }
        actions = {
            name: rng.uniform(low=-0.5, high=0.5, size=shape).astype(np.float32)
            for (name, shape) in action_shapes.items()
        }
        next_states = {
            name: (states[name] + 0.1 * np.sign(states[name])).astype(np.float32)
            for name in state_shapes
        }
        yield states, actions, next_states


def test_torch_reservoir_optimize_and_evaluate():
    rddl = _build_lifted_model()
    learnable = ["EVAPORATION_FACTOR", "RAIN_VAR"]
    batch_size = 2
    data = _reservoir_data_iterator(batch_size, rddl, learnable)
    learner = TorchModelLearner(
        rddl=rddl,
        param_ranges={name: (None, None) for name in learnable},
        batch_size_train=batch_size,
        samples_per_datapoint=1,
        optimizer_kwargs={"lr": 1e-3},
        wrap_non_bool=False,
        logic=_TorchNativeLogic(),
    )
    callback = None
    for callback in learner.optimize_generator(
        data, epochs=3, train_seconds=5.0, print_progress=False
    ):
        pass
    assert callback is not None
    eval_data = itertools.islice(_reservoir_data_iterator(batch_size, rddl, learnable), 5)
    loss_value = learner.evaluate_loss(eval_data, None, callback["param_fluents"])
    assert np.isfinite(loss_value)


if __name__ == "__main__":
    test_torch_reservoir_optimize_and_evaluate()
    print("Torch reservoir smoke test passed.")
