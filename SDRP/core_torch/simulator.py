# ok lets try to make a simulator our algorithm but instead numpy let's use torch
import torch 
from pyRDDLGym.core.simulator import RDDLSimulator 
import torch
import numpy as np
from pyRDDLGym.core.simulator import RDDLSimulator
from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
import pyRDDLGym
import os
from pyRDDLGym.core.simulator import RDDLSimulator
from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.core.compiler.model import RDDLLiftedModel

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

class Simulator(RDDLSimulator):
    def __init__(self, env: RDDLSimulator, device: str = 'cpu'):
        self.env = env
        self.device = device
        self.state = None  # Initialize state to None
    def array2torch(self, action_observation):
        """Convert numpy array to torch tensor."""
        if isinstance(action_observation, dict):
            return {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in action_observation.items()}
        else:
            return torch.tensor(action_observation, dtype=torch.float32, device=self.device)

    def torch2array(self, action_observation):
        """Convert torch tensor to numpy array."""
        if isinstance(action_observation, dict):
            return {k: v.cpu().numpy() for k, v in action_observation.items()}
        else:
            return action_observation.cpu().numpy()


    def sample_action(self):
        """Sample an action from the environment's action space."""
        action = self.env.action_space.sample()
        return action
    
    def sample_reward(self):
        """Sample a reward from the environment's reward space."""
        reward = self.env.sample_reward()
        if isinstance(reward, dict):
            return {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in reward.items()}
        else:
            return torch.tensor(reward, dtype=torch.float32, device=self.device)    


    def sample_observation(self):
        """Sample an observation from the environment's observation space."""
        observation = self.env.observation_space.sample()
        return observation
    
    def step(self, action):
        
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
def main():
    ##
    ##############################################################################################
    #     Create the enviroxnment with config      #
    ##############################################################################################
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #print(f'-------------------base_path: {base_path}')
    domain_rddl   = os.path.join(base_path, "instances", "reservoir", "domain.rddl")
    instance_rddl = os.path.join(base_path, "instances", "reservoir", "instance_1.rddl")

    myEnv = pyRDDLGym.make(
        domain=domain_rddl,
        instance=instance_rddl,
        vectorized=True
    )
    ##############################################################################################
    
    
    # here i have a problem ' if i want ro gradient the action (foe example in race car) 
    # i need to take the tensor for each action but in some cases the action is a dict
    # for some action ( e.g. in reservation tere is only one action - 'release')
    # not only for gar also to convert
    
    print("################################## Actions ##########################################")
    
    # This is just a demo of conversion utilities (does not step an env).
    sim = Simulator(env=myEnv)
    an_action = sim.sample_action()
    #print(f'Sample action from the environment: {an_action}')
    #print ('try to coonver an actionto torch:', sim.array2torch(an_action))
    print(f'Action as numpy array: {an_action}')
    # Convert the action to a torch tensor  
    x = sim.array2torch(an_action)
    print(f'Action as torch tensor: {x}')
    # return x to a numpy array
    x_np = sim.torch2array(x)
    print(f'Action as numpy array from torch: {x_np}')  

    print("################################## Observations ##########################################")
    an_ovservation = sim.sample_observation()
    print(f'Sample observation from the environment: {an_ovservation}')
    # Convert the observation to a torch tensor
    y = sim.array2torch(an_ovservation)
    print(f'Observation as torch tensor: {y}')
    # return y to a numpy array
    y_np = sim.torch2array(y)
    print(f'Observation as numpy array from torch: {y_np}')
    



    print("################################## reward ##########################################")
    an_reward = sim.sample_reward()
    print(f'Sample reward from the environment: {an_reward}')



    print(f'Sample reward from the environment: {an_reward}')
    # Convert the reward to a torch tensor
    z = sim.array2torch(an_reward)
    print(f'Reward as torch tensor: {z}')
    # return z to a numpy array
    z_np = sim.torch2array(z)
    print(f'Reward as numpy array from torch: {z_np}')



    exit()
    print(f'sample an action from the environment: {sim.sample_action()}')
   
    obs_np = {'release': np.array([17.89675 , 15.712295], dtype=np.float32)}
    obs_t = sim.array2torch(obs_np)
    print("Converted to torch:", obs_t)

    back_np = sim.torch2array(obs_t)
    print("Back to numpy:", back_np)


if __name__ == "__main__":
    main()