# SDRP Core Torch README

## Branch Update Workflow
```
(when i want to update yuval-branch 
i push to yuval-branch-2 the updates
git checkout yuval-branch
git pull
git checkout yuval-branch-2 -- README.md      
git add README.md
git commit -m"___"
git push
)
```

## Packege Turorial ( for me )
### Overview
We want to train some againt to solve dome  planing problem( it can work also in rl problem)
for this task we goona use a new package that base on PyRDDLGym_jax but with a few change:
* this package write use pytorch
* splite the algorithm for a few more files.
* __
* __

## Create the environment 
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
# Files 
In this section we present the main files in the pacege 
## main files 
In this packege we have a few main files :
* logic  -  Convert the dynamics from discrete/hybrid to differentiable to make sure we can rollouts the gratient
* Simulator – runs the environment by applying actions to the model and returning the resulting next state & reward  
* Compiler -  Translate RDDL AST into JAX transition/reward functions.
* Planner - to creat a planner and controller
* Model - Gradient-based learning of unknown RDDL parameters(non-fluents) in JAX.
* Tuning - Bayesian hyperparameter tuning for JAX planners via rollouts.
* Planner - Gradient-based planning and policy optimization for RDDL in JAX.

## Logic
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



## Planner

### Main Idea

Gradient-based planning and policy optimization for RDDL in JAX.
### Key classes:
* JaxPlan: \
abstract planner scaffold for losses, optimization loop, and callbacks.
* JaxStraightLinePlan: \
open-loop action sequence optimized via backprop.
* JaxDeepReactivePolicy: \
neural policy mapping states to actions; generalizes SLP.
* GaussianPGPE:\
 Gaussian parameter-exploring policy gradient estimator for policies/trajectories.
* JaxBackpropPlanner: \
orchestrates model compilation, optimization, configs, and training lifecycle.
* JaxOfflineController:\
  optimize, then execute the fixed plan offline at deployment.
* JaxOnlineController:\
 interleave planning and acting during environment execution.
* Preprocessor/StaticNormalizer:\
ormalize and scale actions/states consistently for learning.
* JaxRDDLCompilerWithGrad:\
 compile RDDL into differentiable, gradient-enabled JAX graphs.


## Main functions:
* Parses planner configs,
* compiles differentiable models with JaxRDDLCompilerWithGrad,
* builds open-loop trajectories (JaxStraightLinePlan) or deep reactive policies  (JaxDeepReactivePolicy) using Haiku/Optax,
* applies PGPE/optax optimizers,
* wraps offline/online controllers to optimize then execute plans.



## comiler 
The comiler sets JAX dtypes, initializes values, builds CPF dependency levels, traces objects, and prepares action constraint.
computation graph that can be:

* optimized by XLA,

* differentiated with JAX autograd

* executed in parallel and batched

* used for gradient-based planning and optimization.

This approach produces very high-performance planning, but relies on static graph compilation and exact logical/arithmetical operators.

### Public API
compile (compile invariants/preconditions/terminations/CPFs/reward), compile_transition (step function with optional constraint checks), compile_rollouts (vectorized rollouts under a policy), print_jax (pretty print compiled graphs), model_parameter_info (hyperparameter metadata).
The JaxPlan project compiles RDDL models into symbolic JAX functions.
Each RDDL expression (CPFs, reward, invariants, conditions) is converted into a JAX 


### in our torch
we and to do the nural network for thr DRP more modular like here
Reinforcement Learning (DQN) Tutorial
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


earte file pilicies that contain drp slp 


## Simulator 

### Main Idea

JaxRDDLSimulator extends RDDLSimulator to support the execution of RDDL domains by compiling CPFs, reward functions, and constraints into JAX-compatible functions, and incorporating additional JAX-specific mechanisms.

JaxRDDLSimulator extends RDDLSimulator to support the execution of RDDL domains by compiling CPFs, reward functions, and constraints into JAX-compatible functions, and incorporating additional JAX-specific mechanisms.

### Main functions:
Compiles CPFs/reward/invariants/preconditions/terminations with JaxRDDLCompiler, JITs them, threads a JAX PRNG key through every call, evaluates CPFs in topological order each step, converts JAX tensors back to grounded state/obs dicts, and samples reward/termination.
### Differences from the original simulator:
Uses JAX+XLA instead of eager NumPy, functional PRNG keys instead of in-place RNG, JAX error codes surfaced via handle_error_code instead of immediate Python exceptions, and returns JAX-derived arrays before optional grounding/string conversion.




## model
implements a JAX-based model-learning pipeline:

it defines loss helpers (MSE, BCE, optax wrappers) and the JaxModelLearner class, which compiles an RDDL domain with gradients (JaxRDDLCompilerWithGrad), maps trainable parameters to non-fluent values (with optional range wrapping), builds a JIT’d batched transition + loss, and runs optax-based gradient descent to fit unknown non-fluents from (state, action, next-state) data streams, tracking training status and exporting a learned RDDLLiftedModel.

## Tuning
adds a Bayesian-optimization tuner for JAX planners - 
it defines Hyperparameter (tags with bounds and mapping functions) and JaxParameterTuning, which takes a config template and hyperparameter list(e.g., learning rate, gradient steps, noise scales, rollout counts, horizon, or logic weights.), repeatedly substitutes candidate values, builds/plans with JaxBackpropPlanner, evaluates offline/online returns over multiple trials, and uses multiprocessing plus a GP-based Bayesian optimizer (with optional dashboard logging) to search for the best planner hyperparameters.




## important thing 
when we are update the noise we want to get small noise if the gradient is good** 
and big noise if thr gradient is 0 
the adam update do something that maybe we want to to repeat the adam in leacuter 6 in deep ari talk about it
