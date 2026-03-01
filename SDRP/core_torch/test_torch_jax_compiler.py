 

import os

import jax
import jax.numpy as jnp

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
import torch

from initializer_torch import RDDLValueInitializer

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
print("###################### Parsed RDDL model ########################")

model = RDDLLiftedModel(rddl)

from compiler import TorchRDDLCompiler

reader = RDDLReader(domain_path, instance_path)
text = reader.rddltxt

parser = RDDLParser(lexer=None, verbose=False)
parser.build()
ast = parser.parse(text)              # ast: pyRDDLGym.core.parser.rddl.RDDL

model = RDDLLiftedModel(ast)      
print(model.cpfs)
exit()    # model: RDDLLiftedModel
compiler = TorchRDDLCompiler(model)   # now self exists

compiler.compile()                   # compiles the model, now self.init_values exists
print(compiler.levels)


exit()


# #print(type(reader.rddltxt))  # str
# from collections.abc import Mapping, Sequence

# def dump_ast(obj, name="root", depth=0, max_depth=6, max_items=12, seen=None):
#     if seen is None:
#         seen = set()
#     pad = "  " * depth
#     oid = id(obj)
#     tname = type(obj).__name__

#     if oid in seen:
#         print(f"{pad}{name}: <recursion {tname}>")
#         return
#     seen.add(oid)

#     if depth >= max_depth:
#         print(f"{pad}{name}: {tname} ...")
#         return

#     if isinstance(obj, Mapping):
#         print(f"{pad}{name}: {tname} (len={len(obj)})")
#         for i, (k, v) in enumerate(obj.items()):
#             if i >= max_items:
#                 print(f"{pad}  ...")
#                 break
#             dump_ast(v, name=f"[{k!r}]", depth=depth + 1, max_depth=max_depth, max_items=max_items, seen=seen)
#         return

#     if isinstance(obj, (list, tuple, set)):
#         print(f"{pad}{name}: {tname} (len={len(obj)})")
#         for i, v in enumerate(list(obj)[:max_items]):
#             dump_ast(v, name=f"[{i}]", depth=depth + 1, max_depth=max_depth, max_items=max_items, seen=seen)
#         if len(obj) > max_items:
#             print(f"{pad}  ...")
#         return

#     if hasattr(obj, "__dict__"):
#         fields = vars(obj)
#         print(f"{pad}{name}: {tname} fields={list(fields.keys())}")
#         for i, (k, v) in enumerate(fields.items()):
#             if i >= max_items:
#                 print(f"{pad}  ...")
#                 break
#             dump_ast(v, name=k, depth=depth + 1, max_depth=max_depth, max_items=max_items, seen=seen)
#         return

#     print(f"{pad}{name}: {tname} = {obj!r}")
# print(type(rddl))
# print("top-level:", list(vars(rddl).keys()))   # בדרך כלל: domain, non_fluents, instance, ...
# dump_ast(rddl, max_depth=7, max_items=20)

  # או vars(rddl).keys()
           # RDDL (AST)
#print(type(model))           # RDDLLiftedModel

exit()
# moving to jax values its not happen in the parser and thr RDDLLIftedmodel
#print(model.cpfs)
now_i_check ="rlevel"
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

subs_jax = jax_compiler.init_values
print("##########################################################")
print(subs_jax)
#exit()
#print("##################### subs ########################")

#print(subs)

key = jax.random.PRNGKey(0)


list_of_jax_action_reservior = [ {
    'release': jnp.array(
        [10.453403, 12.082102 ,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
         {
    'release': jnp.array(
        [11.453403, 14.082102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
         {
    'release': jnp.array(
        [10.953403, 11.582102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
         {
    'release': jnp.array(
        [10.253403, 12.782102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
        {
    'release': jnp.array(
        [10.453403, 12.082102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
         {
    'release': jnp.array(
        [11.453403, 14.082102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
         {
    'release': jnp.array(
        [10.953403, 11.582102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} ,
         {
    'release': jnp.array(
        [10.253403, 12.782102,2,3,4,5,6,7,5,4],
        dtype=jnp.float64)} 
]



list_of_jax_actions_race_car = [{
    'fx': jnp.array(-0.12, dtype=jnp.float32),
    'fy': jnp.array(0.32, dtype=jnp.float32)
},
{
    'fx': jnp.array(-0.44, dtype=jnp.float32),
    'fy': jnp.array(0.33, dtype=jnp.float32)
},
{
    'fx': jnp.array(-0.55, dtype=jnp.float32),
    'fy': jnp.array(0.66, dtype=jnp.float32)
}]


jax_counter = 0
model_params = {}  # Example model parameters (if needed)

jax_list_rlvel = []
print("######################################## now for jax ########################################")
for actions_jax in list_of_jax_action_reservior:
    subs_jax, log_jax, model_params_jax=fn_step_jax(key, actions_jax, subs_jax, model_params)
    jax_counter+=1
    print(f"###########################jax counter{jax_counter}########################################")
    subs_jax=subs_jax
    jax_list_rlvel.append(subs_jax[now_i_check])
    #print(subs_jax[now_i_check])
print("#####################jax list rlevel########################")

####################    ##########################################
###################     Torch     ##############################  
##############################################################
from compiler import TorchRDDLCompiler

torch_compiler = TorchRDDLCompiler(model , use64bit =False)
torch_compiler.compile()

print("######################torch compiled successfully########################")

# print("#####################torch################")

# print(torch_compiler.init_values)
# print("#####################jax################")
# print(jax_compiler.init_values) 
# exit()

fn_step_torch = torch_compiler.compile_transition()

key_torch = torch.Generator().manual_seed(0)
subs_torch = dict(torch_compiler.init_values)

# actions_torch = {
#     'fx': torch.tensor(-0.8223206, dtype=torch.float32),
#     'fy': torch.tensor(0.360752, dtype=torch.float32)
# }

list_of_torch_actions_race_car = [{
    'fx': torch.tensor(-0.12, dtype=torch.float32),
    'fy': torch.tensor(0.32, dtype=torch.float32)
},
{
    'fx': torch.tensor(-0.44, dtype=torch.float32),
    'fy': torch.tensor(0.33, dtype=torch.float32)
},
{
    'fx': torch.tensor(-0.55, dtype=torch.float32),
    'fy': torch.tensor(0.66, dtype=torch.float32)
}]

list_of_torch_action_reservior = [ {
    'release': torch.tensor(
        [10.453403, 12.082102 ,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
         {
    'release': torch.tensor(
        [11.453403, 14.082102,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
         {
    'release': torch.tensor(
        [10.953403, 11.582102,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
         {
    'release': torch.tensor(
        [10.253403, 12.782102,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
        {
    'release': torch.tensor(
        [10.453403, 12.082102 ,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
         {
    'release': torch.tensor(
        [11.453403, 14.082102,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
         {
    'release': torch.tensor(
        [10.953403, 11.582102,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} ,
         {
    'release': torch.tensor(
        [10.253403, 12.782102,2,3,4,5,6,7,5,4],
        dtype=torch.float64)} 
]



torch_list_rlvel = []
torc_counter = 0
model_params = {}  # Example model parameters (if needed)


print("######################################## now for torch ########################################")


for actions_torch in list_of_torch_action_reservior:
    subs_torch, log_torch, model_params_torch=fn_step_torch(key_torch, actions_torch, subs_torch, model_params)
    torc_counter+=1
    print(f"####################torch counter{torc_counter}########################################")
    torch_list_rlvel.append(subs_torch[now_i_check])
    subs_torch=subs_torch
    #print(subs_torch[now_i_check])

print("#####################torch list rlevel########################")



import numpy as np


# --- compare Torch vs JAX numerically ---
# Bring JAX array to host as NumPy, then convert to Torch so subtraction is type-compatible.

jax_lists = [np.asarray(jax.device_get(x)).tolist() for x in jax_list_rlvel]
torch_lists = [t.detach().cpu().tolist() for t in torch_list_rlvel]
print(jax_lists[0])
print(torch_lists[0])
for i in range(len(jax_lists)):
    print(f"####################### comparing step {i} ########################")
    diff = "same" if np.all(np.abs(np.array(jax_lists[i]) - np.array(torch_lists[i])) < (0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002)) else "different"
    diff_values = np.abs(np.array(jax_lists[i]) - np.array(torch_lists[i]))
    print(f"diff: {diff}")
    #print(f"diff values: {diff_values}")