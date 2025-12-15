import os
import sys

import numpy as np
import torch
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import core_torch.model as torch_model
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym_jax.core.model import JaxModelLearner


TorchModelLearner = torch_model.TorchModelLearner


class _TorchNativeLogic:
    """Placeholder logic so compiler falls back to raw torch ops."""
    pass
def _load_reservoir_rddl() -> str:
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
    text = f"{domain}\n{instance}"
    print("Reservoir RDDL model under test:\n", text)
    return text


def _build_lifted_model() -> RDDLLiftedModel:
    parser = RDDLParser(None, verbose=False)
    parser.build()
    rddl = parser.parse(_load_reservoir_rddl())
    return RDDLLiftedModel(rddl)
    

def _numpy_non_fluents(model: RDDLLiftedModel, names):
    return {name: np.asarray(model.non_fluents[name], dtype=np.float32) for name in names}


def test_torch_and_jax_model_learners_equivalent():
    model = _build_lifted_model()
    learnable = ["EVAPORATION_FACTOR", "RAIN_VAR"]
    param_ranges = {name: (None, None) for name in learnable}

    torch_learner = TorchModelLearner(
        rddl=model,
        param_ranges=param_ranges,
        batch_size_train=1,
        samples_per_datapoint=1,
        optimizer_kwargs={"lr": 1e-3},
        wrap_non_bool=False,
        logic=_TorchNativeLogic()
    )
    jax_learner = JaxModelLearner(
        rddl=model,
        param_ranges=param_ranges,
        batch_size_train=1,
        samples_per_datapoint=1,
        wrap_non_bool=False
    )

    torch_learner.seed(123)
    base_params = _numpy_non_fluents(model, learnable)

    torch_guess = {
        name: torch.as_tensor(values, dtype=torch_learner.real_dtype)
        for (name, values) in base_params.items()
    }
    torch_params, _ = torch_learner.init_opt_fn(torch_guess)
    torch_param_fluents = torch_learner.map_fn(torch_params)

    jax_guess = {name: jnp.asarray(values) for (name, values) in base_params.items()}
    jax_params, _ = jax_learner.init_opt_fn(jax_guess)
    jax_param_fluents = jax_learner.map_fn(jax_params)

    torch_subs = torch_learner._batched_init_subs()
    jax_subs = jax_learner._batched_init_subs()
    state_name = "rlevel"
    action_name = "release"
    torch_actions = {action_name: torch.full_like(torch_subs[action_name], 0.25)}
    jax_actions = {action_name: jnp.full_like(jax_subs[action_name], 0.25)}

    torch_key = torch.Generator(device="cpu")
    torch_key.manual_seed(999)
    jax_key = jax.random.PRNGKey(999)

    torch_preds, _ = torch_learner.step_fn(
        torch_key,
        torch_param_fluents,
        torch_subs,
        torch_actions,
        torch_learner.compiled.model_params
    )
    jax_preds, _ = jax_learner.step_fn(
        jax_key,
        jax_param_fluents,
        jax_subs,
        jax_actions,
        jax_learner.compiled.model_params
    )

    torch_state = torch_preds[state_name].detach().cpu().numpy()
    jax_state = np.asarray(jax_preds[state_name])
    print("Torch step_fn state:", torch_state)
    print("JAX step_fn state:", jax_state)
    np.testing.assert_allclose(torch_state, jax_state, rtol=1e-6, atol=1e-6)

    target_state = np.zeros_like(torch_state)
    torch_targets = {state_name: torch.as_tensor(target_state, dtype=torch_learner.real_dtype)}
    jax_targets = {state_name: jnp.asarray(target_state)}

    torch_key_loss = torch.Generator(device="cpu")
    torch_key_loss.manual_seed(1234)
    jax_key_loss = jax.random.PRNGKey(1234)

    torch_loss, _ = torch_learner.loss_fn(
        torch_key_loss,
        torch_params,
        torch_subs,
        torch_actions,
        torch_targets,
        torch_learner.compiled.model_params
    )
    jax_loss, _ = jax_learner.loss_fn(
        jax_key_loss,
        jax_params,
        jax_subs,
        jax_actions,
        jax_targets,
        jax_learner.compiled.model_params
    )

    torch_loss_np = torch_loss.detach().cpu().numpy()
    jax_loss_np = np.asarray(jax_loss)
    print("Torch loss:", torch_loss_np)
    print("JAX loss:", jax_loss_np)
    np.testing.assert_allclose(
        torch_loss_np,
        jax_loss_np,
        rtol=1e-6,
        atol=1e-6
    )
if __name__ == "__main__":
    test_torch_and_jax_model_learners_equivalent()
    print("Test passed.")
