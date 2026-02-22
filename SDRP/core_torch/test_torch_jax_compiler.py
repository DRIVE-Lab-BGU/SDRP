 

import os

import jax
import jax.numpy as jnp

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
import torch

from initializer_torch import RDDLValueInitializer

base_path = "/Users/yuvalaroosh/Documents/SDRP/SDRP"
domain_path   = os.path.join(base_path, "instances", "race_car", "domain.rddl")
instance_path = os.path.join(base_path, "instances", "race_car", "instance_1.rddl")
#print(f'-----------domain_path: {domain_path}, instance_path: {instance_path}')
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.compiler.model import RDDLLiftedModel

reader = RDDLReader(domain_path, instance_path)
domain = reader.rddltxt

parser = RDDLParser(lexer=None, verbose=False)
parser.build()
rddl = parser.parse(domain)
print("###################### Parsed RDDL model ########################")


model = RDDLLiftedModel(rddl)

# moving to jax values its not happen in the parser and thr RDDLLIftedmodel
#print(model.cpfs)

##############################################################
###################     Jax     ##############################  
##############################################################
# --- compile jax  ---
jax_compiler = JaxRDDLCompiler(model, use64bit=False )
jax_compiler.compile()
#print("Compiled model successfully.")
# print("Initializing values...")

# print(jax_compiler.init_values)
# exit()

fn_step_jax = jax_compiler.compile_transition()
# the subs in array
subs = dict(jax_compiler.init_values)
#print("##################### subs ########################")

#print(subs)

key = jax.random.PRNGKey(0)


# the action in an array
actions = {'release': jnp.array([  23., 10  ], dtype=jnp.float32)}  # Example action vector
#print("##################### actions ########################")
#print(actions)


model_params = {}  # Example model parameters (if needed)

print("##################### step output ########################")
subs, log, model_params=fn_step_jax(key, actions, subs, model_params)
print(subs)



##############################################################
###################     Torch     ##############################  
##############################################################
from compiler import TorchRDDLCompiler

torch_compiler = TorchRDDLCompiler(model , use64bit =False)
torch_compiler.compile()


print("#####################torch################")

print(torch_compiler.init_values)
print("#####################jax################")
print(jax_compiler.init_values) 
exit()

fn_step_torch = torch_compiler.compile_transition()

key_torch = torch.Generator().manual_seed(0)
subs_torch = dict(torch_compiler.init_values)
actions = {'release': torch.tensor([  23., 10  ], dtype=torch.float32)} 
model_params = {}  # Example model parameters (if needed)
subs_torch, log, model_params=fn_step_torch(key, actions, subs, model_params)
print(subs_torch)


