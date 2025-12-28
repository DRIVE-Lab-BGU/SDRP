# Packege Turorial ( for me )
We want to train some againt to solve dome  planing problem( it can work also in rl problem)
for this task we goona use a new package that base on PyRDDLGym_jax but with a few change:
* this package write use pytorch
* splite the algorithm for a few more files.
* __
* __

# main files 
In this packje we have a few main files :
* logic -the main idea ia create T-norm
* Simulator - to create an envirument to contact
* Compiler -  to make the torch "understend" the envarument
* Planner - to creat a planner and controller

## Train and evaluate Agent
To to it lets deep into the code
* this code take from the original repo pyRDDLGym_jax 
```
import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController

* set up the environment (note the vectorized option must be True)
env = pyRDDLGym.make("domain", "instance", vectorized=True)

################ this line dont exists in the orginal code ##################
* load the config 
planner_args, _, train_args = load_config(config_file)
############################################################################3

* create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
controller = JaxOfflineController(planner, **train_args)

* evaluate the planner ___
controller.evaluate(env, episodes=1, verbose=True, render=True)
env.close()
```
# create the environment 
first we need to create the envitonment of our problem. 
useing 2 files: instnace.rddl & domein.rddl.
the package using this 2  flies by pyRDDLGym.make and create the env.
this envitronment is in numpy and also in normal logict (not the fuzzy logic) .
* pyrddlgym.make -> backend: Type[RDDLSimulator]=RDDLSimulator that defines the logic

Ater we load the config 
###  config file (we wnat to change this process)
This file containe information about:
* model , optimizer , training 
```
[Model]
logic='FuzzyLogic'
comparison_kwargs={'weight': 20}
rounding_kwargs={'weight': 20}
control_kwargs={'weight': 20}

[Optimizer]
method='JaxStraightLinePlan'
method_kwargs={}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': 0.001}

[Training]
key=42
epochs=5000
train_seconds=30
```

When we call  JaxBackpropPlanner we need to give him 2 object
  1) rddl=env.model 
  2) **planner_args
## rddl=env.model
Useing model.py and make the environment to fuzzylogic useing the simulator.py and the comailer.py
simulator.jax -> JaxRDDLCompiler -> from pyRDDLGym_jax.core.logic import ExactLogic (using line 121)
## **planner_args 
_______
# comiler 
JaxPlan Compiler (Original)

The JaxPlan project compiles RDDL models into symbolic JAX functions.
Each RDDL expression (CPFs, reward, invariants, conditions) is converted into a JAX computation graph that can be:

	•	optimized by XLA,

	•	differentiated with JAX autograd,

	•	executed in parallel and batched,

	•	used for gradient-based planning and optimization.

This approach produces very high-performance planning, but relies on static graph compilation and exact logical/arithmetical operators.

Reference:
Gimelfarb, Taitler, Sanner (ICAPS 2024): JaxPlan and GurobiPlan.

⸻

TorchRDDLCompiler (This Project)

This repository implements a Torch-based compiler for RDDL expressions.
The design parallels JaxPlan but is built around:

	•	PyTorch, not JAX

	•	Eager execution, not XLA compilation

	•	Configurable logic (ExactLogic or FuzzyLogic), which can be chosen at the simulator level and   not only at the planner level

The compiler translates the RDDL AST into a collection of pure PyTorch callables, enabling:
	
    •	differentiable simulators with torch.autograd

	•	full visibility into each step of the transition

    •	continuous/fuzzy logical semantics when desired

	•	integration with custom optimization or RL algorithms

	•	compatibility with Torch-based planning pipelines

This makes the Torch compiler ideal for:
	•	fuzzy / soft logic modeling
	•	differentiable optimization of actions
	•	reinforcement learning research
	•	hybrid symbolic–gradient planning

⸻

Key Differences

	•	JaxPlan focuses on speed and static compilation.

	•	TorchRDDLCompiler focuses on flexibility, differentiability, and interpretability.

	•	JaxPlan defaults to ExactLogic; Torch allows ExactLogic or FuzzyLogic interchangeably.

	•	JaxPlan hides execution inside compiled XLA graphs; Torch executes line-by-line in Python.

	•	Torch allows richer custom operators and experimental logic not easily supported in JAX.

# Simulator 


JaxBackpropPlanner get evn.modl and in its change by the logic 


**planner_args go to the model  rddl by hyperparmaters (the config is there )
its mean no more config file back there the user need to write what he wnat 
for example



class dqn(nn.Module):
    def __init__(self):
        super(dqn, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


and then 

policy_net = DQN(n_observations, n_actions).to(device)
and train the policy 

Reinforcement Learning (DQN) Tutorial
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
c

earte file pilicies that contain drp slp 








# logic



# model
to do cliping to the gra



# important thing 
when we are update the noise we want to get small noise if the gradient is good** 
and big noise if thr gradient is 0 
the adam update do something that maybe we want to to repeat the adam in leacuter 6 in deep ari talk about it
