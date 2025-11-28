# Packege Turorial ( for me )
We wnat to train some againt to solve dome  planing problem( it can work also in rl problem)
for this task we goona use a new package that base on PyRDDLGym_jax but with a few change:
* this package write use pytorch
* splite the algorithm for a few more files.
* __
* __

## Train and evaluate Agent
To to it lets deep into the code
```
import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController

* set up the environment (note the vectorized option must be True)
env = pyRDDLGym.make("domain", "instance", vectorized=True)

* create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
controller = JaxOfflineController(planner, **train_args)

* evaluate the planner
controller.evaluate(env, episodes=1, verbose=True, render=True)
env.close()
```
# create envatrument 
first we need to create the envrument of our problem.
useing 2 files: instnace.rddl and domein.rddl.
the package the this 2  flies using pyRDDLGym and create the env.
this env is in numpy way.
but pyrddlgym use boolien logic we want softlogic(defferential -  to backward gradient - update the weghit of the policy(deep))
so we supper from pyrddlgym.simulator and create our simulatore that using soft logic to create the envarument.
pyrddlgym.make -> backend: Type[RDDLSimulator]=RDDLSimulator that defines the logic
simulator.jax -> JaxRDDLCompiler -> from pyRDDLGym_jax.core.logic import ExactLogic (using line 121)
# Simulator 

# logic

# model
