import os
import sys

import pyRDDLGym
import time
from pyRDDLGym_jax.core.planner import (
    JaxBackpropPlanner,
    load_config
)

from SDRP.core.policies import (
    DeterministicJaxPolicy,
    StochasticJaxPolicy
)
from SDRP.core.Logger import Log


class ExampleManager(object):
    def __init__(self, domain, instance, config, policy_type="deterministic", episodes=10, step=10):
        self.domain = domain
        self.instance = instance
        self.config = config
        self.episodes = episodes
        self.step = step
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.rewards = {}
        if policy_type not in ['deterministic', 'stochastic']:
            raise Exception("unknown planner type, planner must be either 'deterministic' or 'stochastic'")
        self.policy_type = policy_type

    def run_example(self,):
        domain_file = os.path.join(self.base_path, "instances", self.domain, "domain.rddl")
        instance_file = os.path.join(self.base_path, "instances", self.domain, self.instance)
        config_file = os.path.join(self.base_path, "configs", self.config)

        myEnv = pyRDDLGym.make(domain=domain_file,
                               instance=instance_file,
                               vectorized=True)

        planner_args, _, train_args = load_config(config_file)
        planner = JaxBackpropPlanner(rddl=myEnv.model, **planner_args)
        if self.policy_type == 'deterministic':
            jax_policy = DeterministicJaxPolicy
        else:
            jax_policy = StochasticJaxPolicy

        rewards = {}
        start_time = time.time()
        for epoch in range(2, self.episodes+1, self.step):
            print(f'pass {time.time() - start_time} seconds')
            print("train on epochs :", epoch)
            agent = jax_policy(planner, **train_args, epochs=epoch)
            # agent = DeterministicJaxPolicy(planner, **train_args, epochs=epoch)
            metrics = agent.evaluate(myEnv, episodes=20)
            rewards[epoch] = metrics['mean']
        print(f'total time {time.time() - start_time} seconds')
        print(rewards)

        self.rewards = rewards

    def log(self, file_name):
        if self.rewards:
            # log = Log(f"log_{self.instance[0:-5]}_sd = 1.csv")
            log = Log(file_name)
            log.logData(self.rewards)

    def get_base_path(self):
        return self.base_path