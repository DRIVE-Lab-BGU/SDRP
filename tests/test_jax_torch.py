import os
import sys
import time

import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxStraightLinePlan, JaxOfflineController
from pyRDDLGym_jax.core.logic import FuzzyLogic, ExactLogic as JaxExactLogic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODULE_ROOT = os.path.join(ROOT, "SDRP")
if MODULE_ROOT not in sys.path and os.path.isdir(MODULE_ROOT):
    sys.path.insert(0, MODULE_ROOT)

from core_torch.planner import TorchBackpropPlanner, TorchStraightLinePlan, TorchOfflineController  # noqa: E402
from core_torch.logic import FuzzyLogic as TorchFuzzyLogic  # noqa: E402

BASE = MODULE_ROOT if os.path.isdir(MODULE_ROOT) else os.path.dirname(os.path.abspath(__file__))
DOMAIN = os.path.join(BASE, "instances", "race_car", "domain.rddl")
INSTANCE = os.path.join(BASE, "instances", "race_car", "instance_3.rddl")



def eval_torch(lr=1e-2, epochs=300, episodes=10, train_seconds=10, batch_size=1):
    env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    plan = TorchStraightLinePlan()
    planner = TorchBackpropPlanner(
        rddl=env.model, plan=plan, logic=TorchFuzzyLogic(),
        batch_size_train=batch_size, rollout_horizon=env.horizon,
        optimizer_kwargs={"lr": lr}, use64bit=False
    )
    ctrl = TorchOfflineController(planner, epochs=epochs, train_seconds=train_seconds, print_summary=True)
    start = time.perf_counter()
    mean = ctrl.evaluate(env, episodes=episodes, verbose=False)["mean"]
    elapsed = time.perf_counter() - start
    return mean, elapsed


def eval_jax(lr=1e-2, epochs=300, episodes=10, train_seconds=60, logic="fuzzy", batch_size=1):
    env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    plan = JaxStraightLinePlan()
    logic_cls = FuzzyLogic if logic == "fuzzy" else JaxExactLogic
    planner = JaxBackpropPlanner(
        rddl=env.model, plan=plan, logic=logic_cls(),
        batch_size_train=batch_size, rollout_horizon=env.horizon,
        optimizer_kwargs={"learning_rate": lr}, pgpe=None
    )
    ctrl = JaxOfflineController(planner, epochs=epochs, train_seconds=train_seconds,
                                print_summary=False, print_progress=False,
                                policy_hyperparams={"release": 1.0})
    start = time.perf_counter()
    mean = ctrl.evaluate(env, episodes=episodes, verbose=False)["mean"]
    elapsed = time.perf_counter() - start
    return mean, elapsed


if __name__ == "__main__":
    try:
        jax_mean, jax_time = eval_jax(logic="fuzzy", batch_size=1)
    except Exception as e:
        jax_mean, jax_time = f"JAX failed: {e}", None
    torch_mean, torch_time = eval_torch(batch_size=1)
    print(f"Torch mean return: {torch_mean} (time {torch_time:.2f}s)")
    if jax_time is not None:
        print(f"JAX   mean return: {jax_mean} (time {jax_time:.2f}s)")
    else:
        print(f"JAX   mean return: {jax_mean}")
