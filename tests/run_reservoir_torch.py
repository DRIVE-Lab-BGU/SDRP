"""Minimal script to solve deterministic reservoir instances with the torch planner.

Usage:
    python run_reservoir_torch.py --instance instance_1.rddl --episodes 1 --epochs 300
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import pyRDDLGym

from core_torch.planner import (
    TorchStraightLinePlan,
    TorchBackpropPlanner,
    TorchOfflineController,
)
from core_torch.logic import FuzzyLogic, ExactLogic


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(base_path: str, instance_name: str):
    domain_file = os.path.join(base_path, "instances", "reservoir", "domain.rddl")
    instance_file = os.path.join(base_path, "instances", "reservoir", instance_name)
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file, vectorized=True)
    return env


def main(instance: str, epochs: int, episodes: int, seed: int, lr: float,
         train_seconds: float) -> None:
    base_path = os.path.dirname(os.path.abspath(__file__))
    set_seeds(seed)
    env = build_env(base_path, instance)
    env.seed(seed)

    plan = TorchStraightLinePlan()
    planner = TorchBackpropPlanner(
        rddl=env.model,
        plan=plan,
        logic=ExactLogic(),  # deterministic reservoir; use exact ops
        batch_size_train=1,
        rollout_horizon=env.horizon,
        optimizer_kwargs={"lr": lr},
        use64bit=False,
    )

    controller = TorchOfflineController(
        planner=planner,
        epochs=epochs,
        train_seconds=train_seconds,
        print_summary=True,
    )

    metrics = controller.evaluate(env, episodes=episodes, verbose=True)
    print(f"Finished. Mean reward: {metrics['mean']:.3f}, std: {metrics['std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch planner demo for reservoir.")
    parser.add_argument("--instance", type=str, default="instance_4.rddl",
                        help="Instance file name inside instances/reservoir/")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--episodes", type=int, default=1, help="Eval episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--train-seconds", type=float, default=30.0,
                        help="Wall-clock budget for training.")
    args = parser.parse_args()
    main(
        instance=args.instance,
        epochs=args.epochs,
        episodes=args.episodes,
        seed=args.seed,
        lr=args.lr,
        train_seconds=args.train_seconds,
    )
