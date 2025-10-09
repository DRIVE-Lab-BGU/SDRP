import sys
import os
import csv

import pyRDDLGym
import pyRDDLGym_jax
import matplotlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from importlib.metadata import version, PackageNotFoundError
# from pyRDDLGym_jax.core.planner import (
# from SDRP.Planner import
from planner import (
    JaxDeepReactivePolicy,
    JaxBackpropPlanner,
    JaxOfflineController,
    load_config
)

##
###############################################
#     Create the environment with config      #
###############################################

base_path = os.path.dirname(os.path.abspath(__file__))

##################
#### for run  ####
##################
instance = "instance_2.rddl"
problem = "reservoir"
sd_for_stoc = 1
pass_for_save = f"log_{instance[0:-5]}_sd = {sd_for_stoc}.csv"


###################################

domain_file = os.path.join(base_path, "instances", problem, "domain.rddl")
instance_file = os.path.join(base_path, "instances", problem, instance)
myEnv = pyRDDLGym.make(domain=domain_file,
                     instance=instance_file,
                     vectorized=True)


# Create the planner for differentiable planning

config_file = os.path.join(base_path, "configs", "Reservoir_DRP.cfg")
planner_args, _, train_args = load_config(config_file)
planner = JaxBackpropPlanner(rddl=myEnv.model, **planner_args)

# The agent wraps the planner and executes training/evaluation
#agent   = JaxOfflineController(planner, **train_args)
#metrics = agent.evaluate(myEnv, episodes=10)
# Measure and print performance metrics
#print(metrics)
#
cum = {}
start_time = time.time()
for epo in range(100,4000,100):
    print(f'pass {time.time() - start_time } seconds')
    print("train on epochs :" , epo)
    agent=  JaxOfflineController(planner, **train_args, epochs = epo , sd=sd_for_stoc)
    # episodes – number of evaluation episodes:
    # For each episode, reset the env (env.reset), roll out the policy until the horizon or done=True
    ## importent!!!!  i change the file policy for evaluate (matrics line 92 sample_action_eval(state) )
    metrics = agent.evaluate(myEnv, episodes=20)
    cum[epo] = metrics['mean'] , metrics['std']
print(f'total time {time.time() - start_time} seconds')
print(cum)

basename = instance


csv_file = os.path.join(base_path, pass_for_save)
write_header = not os.path.exists(csv_file)

# Open the file in append mode
mode='w'
csvfile = open(csv_file, mode=mode, newline='')
writer = csv.DictWriter(csvfile, fieldnames=["epoch", "eval_reward" ,"eval_sd"])
writer.writeheader()

# for r in range(len(cum)):
for _, (key, value) in enumerate(cum.items(), start=0):
    row = {
        'epoch': key,
        'eval_reward': value[0] ,
        'eval_sd': value[1],
    }
    writer.writerow(row)
csvfile.close()




#
#
# plt.figure(figsize=(8, 5))
# plt.plot(list(cum.keys()), list(cum.values()), marker='o')
# plt.xlabel("Epochs")
# plt.ylabel(f"Mean Reward over { 20 } episodes ")
# plt.title(f"Mean Reward vs Epochs,{basename} ,  horizon:{myEnv.horizon}  ")
# plt.grid(True)
# last_x = list(cum.keys())[-1]
# last_y = list(cum.values())[-1]
# plt.text(last_x, last_y, f"{last_y:.2f}", fontsize=10, ha='left', va='bottom')

# plt.show()
sys.exit()
#
# ###############################################
# # Create a policy without using a config file #
# ###############################################
# #Define the policy network (neural network with specified topology)
# plan    = JaxDeepReactivePolicy(topology=[128, 64])
# # Create the planner using the defined policy and optimizer settings
# planner = JaxBackpropPlanner(rddl=myEnv.model, plan=plan,
#                              optimizer_kwargs={'learning_rate': 1e-3},
#                              pgpe=None)
# # Initialize the agent that will train and evaluate the policy
# agent = JaxOfflineController(planner, print_summary=False, train_seconds=20)
# metrics = agent.evaluate(myEnv, episodes=5)
# # Measure and print performance metrics
# print(metrics)


#
# # Save policy parameters (using pickle)
# with open('reservoir_drp.pickle', 'wb') as f:
#     import pickle
#     pickle.dump(agent.params, f)
#
# # Load saved policy parameters and continue evaluation
# with open('reservoir_drp.pickle', 'rb') as f:
#     params = pickle.load(f)
# new_planner = JaxBackpropPlanner(rddl=myEnv.model, plan=JaxDeepReactivePolicy())
# new_agent   = JaxOfflineController(new_planner, params=params, print_summary=False)
# new_agent.evaluate(myEnv, episodes=5)



num_runs = 20
all_rewards = []
for run in range(num_runs):
    state, _ = myEnv.reset()
    cumulative_reward = 0.0
    list_reward = []
#
    for step in range(myEnv.horizon):
        action = agent.sample_action_eval(state)
        action["release"] = action["release"]
        state, reward, *_ = myEnv.step(action)
        cumulative_reward += reward
        list_reward.append(reward)

    print(f'Run {run+1}: cumulative reward = {cumulative_reward}')
    all_rewards.append(list_reward)


all_rewards = np.array(all_rewards)


mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)
basename = instance
# Plot 1: Mean and Smoothed Mean
plt.figure(figsize=(10, 6))
plt.plot(mean_rewards, alpha=0.5, label='Mean Reward (over runs)')
window = 5
rolling_mean = pd.Series(mean_rewards).rolling(window).mean()
plt.fill_between(range(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 color='gray', alpha=0.2, label='±1 std')
plt.plot(rolling_mean, color='orange', label=f'Smoothed mean (window={window})')
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title(f"Average Reward over {num_runs} runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Cumulative Reward
cumulative_all = np.cumsum(all_rewards, axis=1)
cumulative_mean = np.mean(cumulative_all, axis=0)
cumulative_std = np.std(cumulative_all, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_mean, color='green', label='Mean Cumulative Reward')
plt.fill_between(range(len(cumulative_mean)),
                 cumulative_mean - cumulative_std,
                 cumulative_mean + cumulative_std,
                 color='green', alpha=0.2, label='±1 std')
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.title(f"Mean Cumulative Reward over {num_runs} runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Histogram
plt.figure(figsize=(10, 6))
plt.hist(all_rewards.flatten(), bins=20, color='purple', alpha=0.7)
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title(f"Reward Distribution over {num_runs} runs")
plt.grid(True)
plt.show()


################################
# save the plots in the folder #
################################

# basename = instance
# window = 5
# rolling_mean = pd.Series(mean_rewards).rolling(window).mean()
#
# plt.figure(figsize=(10, 6))
# plt.plot(mean_rewards, alpha=0.5, label='Mean Reward (over runs)')
# plt.fill_between(range(len(mean_rewards)),
#                  mean_rewards - std_rewards,
#                  mean_rewards + std_rewards,
#                  color='gray', alpha=0.2, label='±1 std')
# plt.plot(rolling_mean, color='orange', label=f'Smoothed mean (window={window})')
# plt.xlabel("steps")
# plt.ylabel("Reward")
# plt.title(f"Average Reward over {num_runs} runs: {basename} - DRP_no_deterministic_policy")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"{basename}_avg_smoothed_reward_plot_DRP.png")
# plt.close()
# # --- Plot 2: Cumulative Reward (mean over runs) ---
# cumulative_all = np.cumsum(all_rewards, axis=1)  # חישוב מצטבר לכל ריצה
# cumulative_mean = np.mean(cumulative_all, axis=0)
# cumulative_std = np.std(cumulative_all, axis=0)
#
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_mean, color='green', label='Mean Cumulative Reward')
# plt.fill_between(range(len(cumulative_mean)),
#                  cumulative_mean - cumulative_std,
#                  cumulative_mean + cumulative_std,
#                  color='green', alpha=0.2, label='±1 std')
# plt.xlabel("steps")
# plt.ylabel("Cumulative Reward")
# plt.title(f"Mean Cumulative Reward over {num_runs} runs: {basename} - DRP_no_deterministic_policy")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"{basename}_cumulative_reward_plot_avg_DRP.png")
# plt.close()
#
# # --- Plot 3: Histogram of Rewards (across all runs) ---
# plt.figure(figsize=(10, 6))
# plt.hist(all_rewards.flatten(), bins=20, color='purple', alpha=0.7)
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.title(f"Reward Distribution over {num_runs} runs: {basename} - DRP_no_deterministic_policy")
# plt.grid(True)
# plt.savefig(f"{basename}_reward_histogram_avg_DRP.png")
# plt.close()

# state, _ = myEnv.reset()
# cumulative_reward = 0.0
# list_reward = []
# for step in range(myEnv.horizon):
#     action = agent.sample_action_eval(state)
#     action["release"] = action["release"]
#     state, reward, *_ = myEnv.step(action)
#     cumulative_reward += reward
#     list_reward.append(reward)

# print(f'the cumulative reward over   {myEnv.horizon} :  {cumulative_reward}')
# print(f'the cumulative sum over list_reward  {sum(list_reward)}')
# print(f'lisr of rewards {list_reward}')
#
# print(f'lentg of horizon {len(list_reward)}')
# import numpy as np
#
# Plot 2: rewards
# rewards = np.array(list_reward, dtype=float)
# basename = instance
# window = 5
# rolling_mean = pd.Series(rewards).rolling(window).mean()
# plt.figure(figsize=(10, 6))
# plt.plot(rewards, alpha=0.3, label='Raw Reward')
# plt.plot(rolling_mean, color='orange', label=f'Smoothed (window={window})')
# plt.xlabel("steps")
# plt.ylabel(" Reward")
# plt.title(f"Smoothed Reward Over step : {basename} - DRP_no_deterministic_policy")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"{basename}_smoothed_reward_plot_DRP.png")
# plt.close()
#
# # Plot 2: Cumulative
# cumulative = np.cumsum(rewards)
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative, color='green')
# plt.xlabel("steps")
# plt.ylabel("Cumulative Reward")
# plt.title(f"Cumulative Reward Over steps: {basename} - DPR_no_deterministic_policy")
# plt.grid(True)
# plt.savefig(f"{basename}_cumulative_reward_plot_DRP.png")
# plt.close()
#
# # Plot 3: Histogram
# plt.figure(figsize=(10, 6))
# plt.hist(rewards, bins=20, color='purple', alpha=0.7)
# plt.xlabel(" Reward")
# plt.ylabel("Frequency")
# plt.title(f"Reward Distribution Over steps: {basename} - DRP_no_deterministic_policy")
# plt.grid(True)
# plt.savefig(f"{basename}_reward_histogram_DRP.png")
# plt.close()
