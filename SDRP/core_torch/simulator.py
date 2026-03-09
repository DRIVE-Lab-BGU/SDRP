# ok lets try to make a simulator our algorithm but instead numpy let's use torch

import torch
import pyRDDLGym
import os
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
    
from compiler import TorchRDDLCompiler



# TODO:
# 1. init 
# 2. step
# 3. reset
# 4. render ?? 
## 5. close
# 6. seed
x = torch.tensor([114.4, 21.4], requires_grad=True)

# y = x.sum()
# y.backward()

# print(x.grad)

class Simulator():
    def __init__(self, domain_path  , instance_path , state = None , action = None , model_params = None):
        """the init ger paths to the rddl domain and instance 
        files and parse them to create the model"""

        reader = RDDLReader(domain_path, instance_path)
        domain = reader.rddltxt
        parser = RDDLParser(lexer=None, verbose=False)
        parser.build()
        rddl = parser.parse(domain)
        self.model = RDDLLiftedModel(rddl)
        self.horizon = self.model.horizon
        self.torch_compiler = TorchRDDLCompiler(self.model , use64bit =False)
        self.torch_compiler.compile()
        self.fn_step = self.torch_compiler.compile_transition()
        ###
        self.generator = torch.Generator().manual_seed(0)
        ###
        self.state = state
        self.action = action  
        if self.state is None: 
            self.subs_torch = dict(self.torch_compiler.init_values)
        ###
        if self.action is None:
            self.subs_torch.update(dict(self.torch_compiler.init_actions))
        ###    
        if model_params is None:
            self.model_params = dict(self.torch_compiler.init_model_params)
        ###
        pass

    def step(self, action):
        subs_torch, log_torch, model_params_torch=self.fn_step(self.generator, self.action, self.state, self.model_params)
        pass

    def reset(self):
        # Reset the environment and return the initial state
        pass

    def render(self):
        # Render the current state of the environment
        pass

    def close(self):
        # Clean up resources if necessary
        pass

    def seed(self, seed=None):
        # Set the random seed for reproducibility
        pass
    def printer(self):
        print(f"############# horizon : {self.horizon}##############")


def main():

    base_path = "/Users/yuvalaroosh/Documents/SDRP/SDRP"
    domain_path   = os.path.join(base_path, "instances", "reservoir", "domain.rddl")
    instance_path = os.path.join(base_path, "instances", "reservoir", "instance_1.rddl")
    sim = Simulator(domain_path, instance_path)
    print(f'############# horizon : {sim.horizon}    ##############')
    print(f'############# model : {sim.model}    ##############')
    print(f'############# fn step : {sim.fn_step}    ##############')





if __name__ == "__main__":
    main()