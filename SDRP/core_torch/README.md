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




# important thing 
when we are update the noise we want to get small noise if the gradient is good** 
and big noise if thr gradient is 0 
the adam update do something that maybe we want to to repeat the adam in leacuter 6 in deep ari talk about it
