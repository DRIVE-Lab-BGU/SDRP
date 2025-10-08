
import sys
from SDRP.ExampleManager import ExampleManager
import os

# import pyRDDLGym
# import time
# from pyRDDLGym_jax.core.planner import (
#     JaxBackpropPlanner,
#     load_config
# )
#
# from SDRP.policies import (
#     DeterministicJaxPolicy,
#     StochasticJaxPolicy
# )
# from SDRP.Logger import Log

def main(domain, instance, config, episodes=100, step=10):
    manager = ExampleManager(domain, instance, config, episodes, step)
    manager.run_example()
    base_path = manager.get_base_path()
    log_file = os.path.join(base_path, 'logs', f"log_{instance[0:-5]}_sd = 1.csv")
    manager.log(log_file)

# def main(domain, instance, config, episodes=100, step=10):
#     base_path = os.path.dirname(os.path.abspath(__file__))
#     # instance = "instance_2.rddl"
#     # problem = "reservoir"
#
#     domain_file = os.path.join(base_path, "instances", domain, "domain.rddl")
#     instance_file = os.path.join(base_path, "instances", domain, instance)
#     myEnv = pyRDDLGym.make(domain=domain_file,
#                            instance=instance_file,
#                            vectorized=True)
#
#     # Create the planner for differentiable planning
#     # config_file = os.path.join(base_path, "configs", "Reservoir_DRP.cfg")
#     config_file = os.path.join(base_path, "configs", config)
#     planner_args, _, train_args = load_config(config_file)
#     planner = JaxBackpropPlanner(rddl=myEnv.model, **planner_args)
#
#     rewards = {}
#     start_time = time.time()
#     for epoch in range(1, episodes, step):
#         print(f'pass {time.time() - start_time} seconds')
#         print("train on epochs :", epoch)
#         agent = DeterministicJaxPolicy(planner, **train_args, epochs=epoch)
#         # episodes – number of evaluation episodes:
#         # For each episode, reset the env (env.reset), roll out the policy until the horizon or done=True
#         ## importent!!!!  i change the file policy for evaluate (matrics line 92 sample_action_eval(state) )
#         metrics = agent.evaluate(myEnv, episodes=20)
#         rewards[epoch] = metrics['mean']
#     print(f'total time {time.time() - start_time} seconds')
#     print(rewards)
#
#     log = Log(f"log_{instance[0:-5]}_sd = 1.csv")
#     log.logData(rewards)



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python example.py <domain> <instance> <config> [<episodes=100>] [<step=10>]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'config': args[2]}
    if len(args) >= 4: kwargs['episodes'] = int(args[3])
    if len(args) >= 5: kwargs['step'] = int(args[4])
    main(**kwargs)