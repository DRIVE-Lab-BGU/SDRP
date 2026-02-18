# ok lets try to make a simulator our algorithm but instead numpy let's use torch
import torch 
from pyRDDLGym.core.simulator import RDDLSimulator 
import torch
import numpy as np
from pyRDDLGym.core.simulator import RDDLSimulator
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


    def show(self):
        # Display the current state of the environment
        if self.state is not None:
            print("Current state:", self.state)
        else:
            print("State is not initialized.")


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

    # This is just a demo of conversion utilities (does not step an env).
    sim = Simulator(env=None)

    obs_np = {'rlevel': np.array([ 64.42717, 129.9572 ], dtype=np.float32)}
    obs_t = sim.array2torch(obs_np)
    print("Converted to torch:", obs_t)

    back_np = sim.torch2array(obs_t)
    print("Back to numpy:", back_np)


if __name__ == "__main__":
    main()