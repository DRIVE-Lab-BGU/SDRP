"""Minimal one-step examples for JAX compiler transition and torch simulator."""

#from __future__ import annotations

import copy
import sys
from pathlib import Path

import jax
import pyRDDLGym
import torch


ROOT = Path(__file__).resolve().parents[1]
print(f"ROOT={ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler  # noqa: E402
from core_torch.logic import ExactLogic  # noqa: E402
from core_torch.simulator import TorchRDDLSimulator  # noqa: E402
from pyRDDLGym_jax.core import (simulator,
                            model,
                                tuning,
                                logic,
                                compiler,
                                planner)

DOMAIN = ROOT / "instances" / "reservoir" / "domain.rddl"
INSTANCE = ROOT / "instances" / "reservoir" / "instance_1.rddl"
print(f"DOMAIN={DOMAIN}")
print(f"INSTANCE={INSTANCE}")
actions = [ 
 { 'release': [12, 68],} ,
 { 'release': [12, 3],}   ,  
 { 'release': [3, 3],}   ,
 { 'release': [4, 3],}   ,
 { 'release': [5, 3],}   ,   
 { 'release': [6, 3],}   
]

dif_list = {"reward":{"jax": [], "torch": []} , "obs":{"jax": [], "torch": []} }
def jax_single_step(model) -> None:
    print(f"=== JAX compile_transition | {len(actions)} steps ===")
    sim_jax = simulator.JaxRDDLSimulator(model)
    sim_jax.seed(0)
    sim_jax.reset()
    i = 0 
    jax_actions = copy.deepcopy(actions)  # clone the actions to avoid modifying the original list
    for action in jax_actions:
        i += 1
        action['release'] = jax.numpy.array(action['release'])
        obs, reward, done = sim_jax.step(action)

        print(f"the step number{i} observation is {obs}")
        print(f"reward={float(reward)} done={done}")  
        dif_list["reward"]["jax"].append(float(reward))
        dif_list["obs"]["jax"].append(obs)
# we exit here to avoid running the torch simulator, which is not the focus of this test.
def torch_single_step(model) -> None:
    print(f"=== TorchRDDLSimulator.step | {len(actions) } steps ===")
    # here the simulator compiles the model 
    sim = TorchRDDLSimulator(model, logic=ExactLogic() , keep_tensors=True , noise={"type": "smaller_1", "value": [2, 1]} )
    sim.seed(0)
    sim.reset()
    i = 0
    for action in actions:
        i += 1
        obs, reward, done = sim.step(action, i)

        print(f"the step number{i} observation is {obs}")
        print(f"reward={float(reward)} done={done}")
        dif_list["reward"]["torch"].append(float(reward))
        dif_list["obs"]["torch"].append(obs)
def main() -> None:
    #################################################################################
    ###### we can also run the torch simulator using the lifted model directly ######

    #from pyRDDLGym.core.parser.reader import RDDLReader
    #from pyRDDLGym.core.parser.parser import RDDLParser
    #from pyRDDLGym.core.compiler.model import RDDLLiftedModel
    #reader = RDDLReader(DOMAIN, INSTANCE)
    #domain = reader.rddltxt
    #parser = RDDLParser(lexer=None, verbose=False)
    #parser.build()
    #rddl = parser.parse(domain)
    #model_no_ptRDDLGym = RDDLLiftedModel(rddl)
    #torch_single_step(model_no_ptRDDLGym)
    #####################################################################################

    env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    model = env.model
    torch_single_step(model)
    exit(0)  # we exit here to avoid running the torch simulator, which is not the focus of this test.
    jax_single_step(model)
    torch_single_step(model)
    import numpy as np
    
    print(f"reward difference: {np.array(dif_list['reward']['jax']) - np.array(dif_list['reward']['torch'])}")


    obs_diff = []
    for jax_obs, torch_obs in zip(dif_list["obs"]["jax"], dif_list["obs"]["torch"]):
        diff_t1 = float(jax_obs["rlevel___t1"]) - float(torch_obs["rlevel"][0])
        diff_t2 = float(jax_obs["rlevel___t2"]) - float(torch_obs["rlevel"][1])
        obs_diff.append([diff_t1, diff_t2])

    print("observation difference:")
    print(np.array(obs_diff))
if __name__ == "__main__":
    main()
