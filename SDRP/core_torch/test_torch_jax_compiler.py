 

import os

import jax
import jax.numpy as jnp

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
import torch

base_path = "/Users/yuvalaroosh/Documents/SDRP/SDRP"
domain_path   = os.path.join(base_path, "instances", "reservoir", "domain.rddl")
instance_path = os.path.join(base_path, "instances", "reservoir", "instance_1.rddl")
#print(f'-----------domain_path: {domain_path}, instance_path: {instance_path}')
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.compiler.model import RDDLLiftedModel

reader = RDDLReader(domain_path, instance_path)
domain = reader.rddltxt

parser = RDDLParser(lexer=None, verbose=False)
parser.build()
rddl = parser.parse(domain)

model = RDDLLiftedModel(rddl)

# print(f'discount: {model.discount}, horizon: {model.horizon}, cpfs: {model.cpfs.keys()}')
# print(" #######. cpfs ##########")
# print(f'initial state: {model.cpfs}')
# print(" #######. reward ##########")
# print(f'reward: {model.reward}')
# print("Parsed model successfully.")
##############################################################
###################     Jax     ##############################  
##############################################################
# --- compile jax  ---
jax_compiler = JaxRDDLCompiler(model, use64bit=False )
jax_compiler.compile()
#print("Compiled model successfully.")


fn_step_jax = jax_compiler.compile_transition()
# the subs in array
subs = dict(jax_compiler.init_values)
#print("##################### subs ########################")

#print(subs)

key = jax.random.PRNGKey(0)
print(key)

# the action in an array
actions = {'release': jnp.array([  23., 10  ], dtype=jnp.float32)}  # Example action vector
#print("##################### actions ########################")
#print(actions)


model_params = {}  # Example model parameters (if needed)

#print("##################### step output ########################")
subs, log, model_params=fn_step_jax(key, actions, subs, model_params)
#print(subs)



##############################################################
###################     Torch     ##############################  
##############################################################
from compiler import TorchRDDLCompiler

torch_compiler = TorchRDDLCompiler(model , use64bit =False)
torch_compiler.compile()
fn_step_torch = torch_compiler.compile_transition()

key_torch = torch.Generator().manual_seed(0)
subs_torch = dict(torch_compiler.init_values)
actions = {'release': jnp.array([  23., 10  ], dtype=jnp.float32)} 
model_params = {}  # Example model parameters (if needed)
checkiftensor = torch_compiler.convert2torch(actions)
print(f'Is the converted action a torch tensor? {isinstance(checkiftensor, torch.Tensor)}')

exit()
subs_torch, log, model_params=fn_step_torch(key, actions, subs, model_params)
print(subs_torch)


