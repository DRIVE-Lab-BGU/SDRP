(when i want to update yuval-branch 
i push to yuval-branch-2 the updates
git checkout yuval-branch
git pull
git checkout yuval-branch-2 -- README.md      
git add README.md
git commit -m"___"
git push
)

# Packege Turorial ( for me )
We want to train some againt to solve dome  planing problem( it can work also in rl problem)
for this task we goona use a new package that base on PyRDDLGym_jax but with a few change:
* this package write use pytorch
* splite the algorithm for a few more files.
* __
* __

# main files 
In this packje we have a few main files :
* logic  -  Convert the dynamics from discrete/hybrid to differentiable to make sure we can rollouts the gratient
* Simulator – runs the environment by applying actions to the model and returning the resulting next state & reward  
* Compiler -  to make the torch "understend" the envarument
* Planner - to creat a planner and controller

## Train and evaluate Agent
To to this lets deep into the code:
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
useing 2 files: 
* instnace.rddl 
* domein.rddl \
the package using this 2  flies by pyRDDLGym.make and create the env. \
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
# Logic
### Main Idea
Replace every discrete / non-differentiable logical operator with a smooth, parameterized function, so the entire RDDL model becomes differentiable end-to-end, and those parameters can be optimized with gradients.  

implements a plug‑and‑play “logic backend” for pyRDDLGym in JAX, offering exact Boolean logic and smooth/differentiable fuzzy relaxations plus reparameterized sampling for discrete distributions.
* plug-and-play denotes a modular architecture in which alternative logic backends (e.g., exact Boolean semantics or differentiable fuzzy relaxations with reparameterized sampling) can be interchanged transparently, as they conform to a common Logic interface and operator dictionary expected by the compiler.
### Main Function
* Comparison & rounding: 

 Comparison/SigmoidComparison produce sigmoid-based ≷/==/sgn/argmax approximations with learnable slope stored in init_params[id]. Rounding/SoftRounding provide smooth floor/round (softfloor/softround) with tunable sharpness.
* Negation & t-norms: 

 Complement/StandardComplement implement 1–x. TNorm plus implementations (ProductTNorm, GodelTNorm, LukasiewiczTNorm, YagerTNorm) define fuzzy AND and forall aggregation variants; Yager uses an Lp-style norm with parameter p saved in init_params.
* Random sampling: 

RandomSampling interface with two concrete implementations. SoftRandomSampling uses differentiable relaxations: Discrete via Gumbel-softmax; Poisson via exponential thinning or truncated Gumbel-softmax with normal fallback for large rates; Binomial via truncated Gumbel-softmax or normal; NegativeBinomial via Gamma–Poisson mixture; Geometric via inverse-CDF with soft floor; Bernoulli via uniform threshold or optional Gumbel-softmax. Determinization simply returns means/expectations instead of sampling.

### class
* Logic:

 base sets JAX precision (32/64) and declares abstract ops (logical, comparison, rounding, indexing, control, sampling). get_operator_dicts exposes operator name → callable mappings used by the compiler. Each concrete op factory returns a JAX-callable along with a mutable params dict that carries the initialized hyperparameters.
* Exact vs fuzzy:

ExactLogic wires everything to crisp JAX ops/random samplers (including tfp NegativeBinomial). FuzzyLogic composes the above fuzzy components: OR/exists/not-equal, etc., are built from t-norm + complement; sqrt/div/mod/ceil add small stabilizers; argmin is derived from argmax. Hyperparameters (tnorm, complement, comparison weights, rounding sharpness, sampling strategy, control softness, eps, 64-bit) are configurable via the constructor/string summary.



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

### in our torch
we and to do the nural network for thr DRP more modular like here
Reinforcement Learning (DQN) Tutorial
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


earte file pilicies that contain drp slp 


# Simulator 

### Main Idea

JaxRDDLSimulator extends RDDLSimulator to support the execution of RDDL domains by compiling CPFs, reward functions, and constraints into JAX-compatible functions, and incorporating additional JAX-specific mechanisms.

JaxRDDLSimulator extends RDDLSimulator to support the execution of RDDL domains by compiling CPFs, reward functions, and constraints into JAX-compatible functions, and incorporating additional JAX-specific mechanisms.

### Main functions:
Compiles CPFs/reward/invariants/preconditions/terminations with JaxRDDLCompiler, JITs them, threads a JAX PRNG key through every call, evaluates CPFs in topological order each step, converts JAX tensors back to grounded state/obs dicts, and samples reward/termination.
### Differences from the original simulator:
Uses JAX+XLA instead of eager NumPy, functional PRNG keys instead of in-place RNG, JAX error codes surfaced via handle_error_code instead of immediate Python exceptions, and returns JAX-derived arrays before optional grounding/string conversion.




# model
to do cliping to the gra



# important thing 
when we are update the noise we want to get small noise if the gradient is good** 
and big noise if thr gradient is 0 
the adam update do something that maybe we want to to repeat the adam in leacuter 6 in deep ari talk about it
