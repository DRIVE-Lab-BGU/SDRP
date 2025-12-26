import os
import sys
import time
import argparse
from typing import List, Tuple, Optional

import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController, JaxStraightLinePlan
from pyRDDLGym_jax.core.logic import FuzzyLogic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODULE_ROOT = os.path.join(ROOT, "SDRP")
if MODULE_ROOT not in sys.path and os.path.isdir(MODULE_ROOT):
    sys.path.insert(0, MODULE_ROOT)

from core_torch.planner import TorchBackpropPlanner, TorchOfflineController, TorchStraightLinePlan  # noqa: E402
from core_torch.logic import FuzzyLogic as TorchFuzzyLogic  # noqa: E402


def eval_torch(domain_file: str, instance_file: str,
               lr: float = 1e-2, epochs: Optional[int] = 50, episodes: int = 3,
               train_seconds: Optional[float] = 5.0, batch_size: int = 4) -> Tuple[float, float]:
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file, vectorized=True)
    plan = TorchStraightLinePlan()
    planner = TorchBackpropPlanner(
        rddl=env.model,
        plan=plan,
        logic=TorchFuzzyLogic(),
        batch_size_train=batch_size,
        rollout_horizon=env.horizon,
        optimizer_kwargs={"lr": lr},
        use64bit=False,
    )
    ctrl = TorchOfflineController(
        planner,
        epochs=epochs,
        train_seconds=train_seconds,
        print_summary=False,
    )
    start = time.perf_counter()
    mean = ctrl.evaluate(env, episodes=episodes, verbose=False)["mean"]
    elapsed = time.perf_counter() - start
    return mean, elapsed


def eval_jax(domain_file: str, instance_file: str,
             lr: float = 1e-2, epochs: Optional[int] = 50, episodes: int = 3,
             train_seconds: Optional[float] = 5.0, batch_size: int = 4) -> Tuple[float, float]:
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file, vectorized=True)
    plan = JaxStraightLinePlan()
    planner = JaxBackpropPlanner(
        rddl=env.model,
        plan=plan,
        logic=FuzzyLogic(),
        batch_size_train=batch_size,
        rollout_horizon=env.horizon,
        optimizer_kwargs={"learning_rate": lr},
        pgpe=None,
    )
    ctrl = JaxOfflineController(
        planner,
        epochs=epochs,
        train_seconds=train_seconds,
        print_summary=False,
        print_progress=False,
        policy_hyperparams={name: 1.0 for name in env.model.action_fluents},
    )
    start = time.perf_counter()
    mean = ctrl.evaluate(env, episodes=episodes, verbose=False)["mean"]
    elapsed = time.perf_counter() - start
    return mean, elapsed


def list_instances(domain_dir: str) -> List[str]:
    return sorted(
        f for f in os.listdir(domain_dir)
        if f.endswith(".rddl") and f.startswith("instance")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-seconds", type=float, default=5.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    scenarios = [
        ("reservoir", list_instances(os.path.join(MODULE_ROOT, "instances", "reservoir"))),
        ("race_car", list_instances(os.path.join(MODULE_ROOT, "instances", "race_car"))),
    ]
    results = []
    for domain_name, inst_list in scenarios:
        domain_file = os.path.join(MODULE_ROOT, "instances", domain_name, "domain.rddl")
        # race_car has more complex CPF shapes; use batch=1 to avoid broadcast issues
        batch_size = 1 if domain_name == "race_car" else 4
        for inst in inst_list:
            instance_file = os.path.join(MODULE_ROOT, "instances", domain_name, inst)
            print(f"\n=== {domain_name}/{inst} ===")
            try:
                t_mean, t_time = eval_torch(domain_file, instance_file,
                                            lr=args.lr, epochs=args.epochs,
                                            episodes=args.episodes,
                                            train_seconds=args.train_seconds,
                                            batch_size=batch_size)
                print(f"Torch mean: {t_mean:.4f} (time {t_time:.2f}s)")
            except Exception as e:
                t_mean, t_time = f"Torch failed: {e}", None
                print(t_mean)
            try:
                j_mean, j_time = eval_jax(domain_file, instance_file,
                                          lr=args.lr, epochs=args.epochs,
                                          episodes=args.episodes,
                                          train_seconds=args.train_seconds,
                                          batch_size=batch_size)
                print(f"JAX   mean: {j_mean:.4f} (time {j_time:.2f}s)")
            except Exception as e:
                j_mean, j_time = f"JAX failed: {e}", None
                print(j_mean)
            results.append((domain_name, inst, t_mean, t_time, j_mean, j_time))
    print("\nSummary:")
    for dom, inst, tm, tt, jm, jt in results:
        print(f"{dom}/{inst}: Torch={tm} ({tt}) | JAX={jm} ({jt})")


if __name__ == "__main__":
    main()
