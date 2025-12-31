#  Core jax README

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

## Packege Turorial 
### Overview
We want to train some controller to solve some planing problem, 
for this task we going use a new package that base on PyRDDLGym_jax but with a few change:
* this package write use pytorch
* splite the algorithm for a few more files.
* __
* __
# Code 
The total code to solve a planing probelm is: 
```
import pyRDDLGym
from pyRDDLGym_jax.core.planner import (JaxBackpropPlanner, JaxOfflineController)

env = pyRDDLGym.make("domain", "instance", vectorized=True)

planner_args, _, train_args = load_config(config_file)

planner = JaxBackpropPlanner(rddl=env.model, **planner_args)

controller = JaxOfflineController(planner, **train_args) 

controller.evaluate(env, episodes=1, verbose=True, render=True)

env.close()
```
lets deep inside the code: 


## Create the environment 
```
# set up the environment (note the vectorized option must be True)
env = pyRDDLGym.make("domain", "instance", vectorized=True)
```
The first step in training the controller is to construct an environment.

This environment is defined using two RDDL files:

domain.rddl — specifies the domain dynamics, predicates, actions, CPFs, constraints, and reward.

instance.rddl — defines a concrete problem instance, including objects and initial state.

These two files are loaded using pyRDDLGym.make, which constructs the environment from the RDDL specification.

The resulting environment operates in NumPy and uses standard (exact) logic, rather than fuzzy logic

###  config file 
```
planner_args, _, train_args = load_config(config_file)
```
This file containe information about: 
* Model - Picks logic backend and its hyperparameters.

* Optimizer - Chooses planning method and gradient optimizer settings.

* Training - Sets seeds, epochs, and time budget.
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


instantiates a gradient-based planner

## Create the Planner

```
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
```
When we call  JaxBackpropPlanner we need to give him 2 object
  1) rddl=env.model 
  2) **planner_args


rddl=env.model: supplies the standard(ExactLogic) RDDL model (with its CPFs, in Numpy)

 planner_args (from the config) - tell JaxBackpropPlanner to compile it into the fuzzy-logic, differentiable version used for planning.
## Create the Controller & Train
```
controller = JaxOfflineController(planner, **train_args) 
```
Creates an offline controller that leverages the planner to select and execute actions while interacting with the environment, using the training settings specified in the config.

## Evaluate 
```
controller.evaluate(env, episodes=1, verbose=True, render=True)
env.close()
``` 
Runs the controller (using the planner’s policy) for one episode in the environment with logging/rendering, then closes the environment.



# Files 
This section provides an overview of the core files in the package and their functionality.

### main files 
* logic  -  Convert the dynamics from discrete/hybrid to differentiable to make sure we can rollouts the gratient

* Simulator – runs the environment by applying actions to the model and returning the resulting next state & reward  

* Compiler -  Translate RDDL AST into JAX transition/reward functions.

* Planner - Gradient-based planning and policy optimization for RDDL in JAX.


* Model - Gradient-based learning of unknown RDDL parameters
(non-fluents) in JAX.

* Tuning - Bayesian hyperparameter tuning for JAX planners via rollouts.




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

base sets JAX precision (32/64) and declares abstract ops (logical, comparison, rounding, indexing, control, sampling). 
get_operator_dicts exposes operator name → callable mappings used by the compiler. Each concrete op factory returns a JAX-callable along with a mutable params dict that carries the initialized hyperparameters.
* Exact vs fuzzy:

ExactLogic wires everything to crisp JAX ops/random samplers (including tfp NegativeBinomial). FuzzyLogic composes the above 
fuzzy components: OR/exists/not-equal, etc., are built from t-norm + complement; sqrt/div/mod/ceil add small stabilizers; argmin is derived from argmax. Hyperparameters (tnorm, complement, comparison weights, rounding sharpness, sampling strategy, control softness, eps, 64-bit) are configurable via the constructor/string summary.



## Simulator 

### Main Idea


JaxRDDLSimulator extends RDDLSimulator to support the execution of RDDL domains by compiling CPFs, reward functions, and constraints into JAX-compatible functions, and incorporating additional JAX-specific mechanisms.

### Main functions:
Compiles CPFs/reward/invariants/preconditions/terminations with JaxRDDLCompiler, JITs them, threads a JAX PRNG key through every call, evaluates CPFs in topological order each step, converts JAX tensors back to grounded state/obs dicts, and samples reward/termination.
### Differences from the original simulator:
Uses JAX+XLA instead of eager NumPy, functional PRNG keys instead of in-place RNG, JAX error codes surfaced via handle_error_code instead of immediate Python exceptions, and returns JAX-derived arrays before optional grounding/string conversion.


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
 normalize and scale actions/states consistently for learning.
* JaxRDDLCompilerWithGrad:\
 compile RDDL into differentiable, gradient-enabled JAX graphs.


### Main functions:
* Parses planner configs,
* compiles differentiable models with JaxRDDLCompilerWithGrad,
* builds open-loop trajectories (JaxStraightLinePlan) or deep reactive policies  (JaxDeepReactivePolicy) using Haiku/Optax,
* applies PGPE/optax optimizers,
* wraps offline/online controllers to optimize then execute plans.





## model
implements a JAX-based model-learning pipeline:

it defines loss helpers (MSE, BCE, optax wrappers) and the JaxModelLearner class, which compiles an RDDL domain with gradients (JaxRDDLCompilerWithGrad), maps trainable parameters to non-fluent values (with optional range wrapping), builds a JIT’d batched transition + loss, and runs optax-based gradient descent to fit unknown non-fluents from (state, action, next-state) data streams, tracking training status and exporting a learned RDDLLiftedModel.

## Tuning
adds a Bayesian-optimization tuner for JAX planners - 
it defines Hyperparameter (tags with bounds and mapping functions) and JaxParameterTuning, which takes a config template and hyperparameter list(e.g., learning rate, gradient steps, noise scales, rollout counts, horizon, or logic weights.), repeatedly substitutes candidate values, builds/plans with JaxBackpropPlanner, evaluates offline/online returns over multiple trials, and uses multiprocessing plus a GP-based Bayesian optimizer (with optional dashboard logging) to search for the best planner hyperparameters.


## Flow procces through the package 
Here is the end-to-end flow for training a controller with `pyRDDLGym_jax`:

```
domain.rddl + instance.rddl
        │
        ▼
pyRDDLGym.make(..., vectorized=True)  ➜  env.model (RDDL graph)
        │
        ▼
load_config(config_file)
│        ├─ planner_args  (logic backend, optimizer, horizons…)
│        └─ train_args    (epochs, time budget, seeds…)
        │
        ▼
JaxBackpropPlanner(rddl=env.model, **planner_args)
│        ├─ compiles fuzzy/exact models (JaxRDDLCompilerWithGrad / JaxRDDLCompiler)
│        ├─ builds JaxPlan (default JaxStraightLinePlan: per-step action params)
│        └─ constructs differentiable rollouts + loss = -utility(returns)
        │
        ▼
JaxOfflineController(planner, **train_args)
│        └─ train(): planner.optimize(...)
│              ├─ _jax_init → initialize plan params + optax optimizer state
│              └─ for each epoch / until time budget:
│                    loss, log = train_rollouts(...)          # forward pass
│                    grad = jax.value_and_grad(loss)(...)     # compute ∂loss/∂actions
│                    updates = optimizer.update(grad, state)
│                    params = optax.apply_updates(params, updates)
│                    params = plan.projection(...)/clamp to bounds
        │
        ▼
controller.evaluate(env, episodes=..., render=...)
│        └─ uses learned params via planner.test_policy to act in env
```

Gradient update location in code: `SDRP/planner.py:2176-2214` (`JaxBackpropPlanner._jax_update`) where `jax.value_and_grad` computes the gradient and `optax.update/optax.apply_updates` adjust the action parameters, followed by projection to respect action bounds.




# important thing 
* when we are update the noise we want to get small noise if the gradient is good** 
and big noise if thr gradient is 0 
the adam update do something that maybe we want to to repeat the adam in leacuter 6 in deep ari talk about it

* in our torchwe and to do the nural network for thr DRP more modular like here
Reinforcement Learning (DQN) Tutorial
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
earte file pilicies that contain drp slp 

* we need to change the config because we wnat the nn out / and maybe we dont have a tuning 


# How to work
## create Simulator and check torch vs jax
the simulator is takes from pyRDDLGym , the change ti fuzzylogic happen in the planner in the JaxBackpropPlanner.


        │
        ▼
## create the logic
he conversion is straightforward.
check it using a test file, check evrey function (convert)

        │
        ▼
## Create simulator file
handle_error_code - bilud in the comiler (bilud the same way)
in jax simulator there is a comilpe rddl so :

        │
        ▼
## create torch compiler 
This part is significantly different, because the planner is implemented in PyTorch and all updates are performed eagerly.


        │
        ▼
## create planner - JaxPlan

        │
        ▼

## create planner - JaxBackpropPlanner 
here we wnat to check the simulator after convert to fuzzy logic.
so create JaxBackpropPlanner 
and to file test check the simulator.

        │
        ▼
## create planner - JaxRDDLCompilerWithGrad

Compiles an RDDL model into fully differentiable transition and reward functions by replacing discrete operations with smooth approximations.

> add the adaptive noise (for now only sd =0 )

Conversion approach:

In the Torch version, all operators are reimplemented using eager PyTorch tensors and autograd, with non-differentiable components handled via detach() where needed.

        │
        ▼
## create planner - JaxOfflineController 

Conversion approach:

The Torch version preserves the same control logic, replacing JAX PRNG handling with a Torch generator while keeping all updates eager.

        │
        ▼
## create planner - Preprocessor/StaticNormalizer 
Applies static min–max normalization to state variables based on bounds inferred from the RDDL domain.

Conversion approach:

In Torch, the same normalization logic is implemented with eager tensor operations, without JIT compilation or graph transformations.

        │
        ▼
## create planner - JaxOnlineController
we dont need it?


        │
        ▼
## create planner - SLP
Verify this using a test file on several deterministic and stochastic domains.

        │
        ▼
## create planner-DRP
 we wnat the user create the network in this file and the DRP will get the user nn.
 like here: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 

Verify this using a test file on several deterministic and stochastic domains.



# Important Notes

>When updating the noise parameters, we want the noise magnitude to adapt to the gradient quality:
small noise when the gradient signal is strong, and larger noise when the gradient is close to zero.
This behavior is similar to what Adam implicitly achieves through its adaptive step sizes, and we may want to explicitly replicate the mechanism discussed in Lecture 6 of the Deep ARI course.

>The configuration format needs to be updated: the neural network should be defined externally by the user, and hyperparameter tuning may be disabled or handled separately.