import math
import os
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
# ensure project root is on sys.path when running this file directly
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# project files live under ROOT / "SDRP"
PROJECT_DIR = ROOT / "SDRP"
if not PROJECT_DIR.is_dir():
    PROJECT_DIR = ROOT
MODULE_ROOT = PROJECT_DIR
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
DOMAIN = PROJECT_DIR / "instances" / "reservoir" / "domain.rddl"
INSTANCE = PROJECT_DIR / "instances" / "reservoir" / "instance_1.rddl"

import pyRDDLGym  # noqa: E402
from pyRDDLGym.core.simulator import RDDLSimulator  # noqa: E402
from core_torch.simulator import TorchRDDLSimulator  # noqa: E402
from core_torch.logic import ExactLogic  # noqa: E402


@pytest.fixture(scope="module")
def env():
    try:
        return pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    except Exception as exc:
        pytest.skip(f"Could not load local reservoir files via pyRDDLGym: {exc}")


def _rollout(sim, horizon):
    obs, rewards = [], []
    actions = sim.noop_actions
    for _ in range(horizon):
        state, reward, done = sim.step(actions)
        obs.append(state)
        rewards.append(reward)
        if done:
            break
    return obs, rewards


def test_torch_matches_numpy_simulator(env):
    """Deterministic reservoir: torch simulator should match numpy simulator for noop actions."""
    horizon = min(5, env.horizon)

    # torch simulator
    torch_sim = TorchRDDLSimulator(env.model, logic=ExactLogic())
    torch_sim.seed(0)
    torch_obs, torch_rewards = _rollout(torch_sim, horizon)

    # numpy baseline using same noop actions converted to numpy
    np_sim = RDDLSimulator(env.model, logger=None)
    np_actions = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
        for (k, v) in torch_sim.noop_actions.items()
    }
    np_sim.reset()
    np_obs, np_rewards = [], []
    for _ in range(horizon):
        np_sim.check_action_preconditions(np_actions)
        state, reward, _ = np_sim.step(np_actions)
        np_obs.append(state)
        np_rewards.append(reward)

    assert len(np_rewards) == len(torch_rewards)
    for r_np, r_torch in zip(np_rewards, torch_rewards):
        assert math.isfinite(r_torch)
        np.testing.assert_allclose(r_torch, r_np, atol=1e-5)

    # compare first state's rlevel fluent across simulators
    rlevel_key = next(k for k in np_obs[0].keys() if "rlevel" in k)
    np_rlevel = np_obs[0][rlevel_key]
    torch_rlevel = torch_obs[0][rlevel_key]
    np.testing.assert_allclose(torch_rlevel, np_rlevel, atol=5e-2)


@pytest.mark.skip(reason="JAX not reliably importable in this environment")
def test_torch_matches_jax_simulator(env):
    """Optional: compare against Jax simulator if available."""
    from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator

    horizon = min(5, env.horizon)

    torch_sim = TorchRDDLSimulator(env.model, logic=ExactLogic())
    torch_sim.seed(0)
    torch_obs, torch_rewards = _rollout(torch_sim, horizon)

    jax_sim = JaxRDDLSimulator(env.model, raise_error=True, keep_tensors=True)
    jax_sim.seed(0)

    # adapter because JaxRDDLSimulator.step returns state only
    jax_states = []
    jax_rewards = []
    actions = jax_sim.noop_actions
    for _ in range(horizon):
        state = jax_sim.step(actions)
        reward = jax_sim.sample_reward()
        jax_states.append(state)
        jax_rewards.append(reward)

    for r_torch, r_jax in zip(torch_rewards, jax_rewards):
        np.testing.assert_allclose(r_torch, r_jax, atol=1e-5)
    rlevel_key = next(k for k in env.model.state_fluents if "rlevel" in k)
    np.testing.assert_allclose(
        torch_obs[0][rlevel_key], jax_states[0][rlevel_key], atol=1e-5
    )
