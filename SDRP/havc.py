import os
import pyRDDLGym
from pyRDDLGym.core.policy import RandomPolicy
from pyRDDLGym.core.visualizer import RDDLVisualizer

# ===== paths =====
base_path = os.path.dirname(os.path.abspath(__file__))

domain_file = os.path.join(base_path, "instances", "havc", "domain.rddl")
instance_file = os.path.join(base_path, "instances", "havc", "instance2.rddl")

# ===== create environment =====
env = pyRDDLGym.make(
    domain_file=domain_file,
    instance_file=instance_file
)

# ===== weak policy (random) =====
policy = RandomPolicy(env.model)

# ===== visualizer =====
viz = RDDLVisualizer(env.model)

# ===== rollout =====
state = env.reset()
frames = []

horizon = 50  # קצר! רק לראות תנועה

for t in range(horizon):
    action = policy.sample_action(state)
    state, reward, done, info = env.step(action)

    frame = viz.render(state)
    frames.append(frame)

    if done:
        break

env.close()

# ===== save GIF =====
gif_path = "havc_demo.gif"
viz.save_gif(frames, gif_path, fps=5)

print(f"GIF saved to {gif_path}")