####
import sys
import os
import csv
import imageio
import pyRDDLGym
# form pyRDDLGym.core.compiler import RDDLLiftedModel
from IPython.display import Image
import os

import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent
from pyRDDLGym.core.visualizer.movie import MovieGenerator


from pyRDDLGym_jax.core import logic
import matplotlib
from pyRDDLGym.core import env

from pyRDDLGym_jax.core import (simulator,
                            model,
                                tuning,
                                logic,
                                compiler,
                                planner)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time




from importlib.metadata import version, PackageNotFoundError
# from pyRDDLGym_jax.core.planner import (
# from SDRP.Planner import
from planner import (
    JaxDeepReactivePolicy,
    JaxBackpropPlanner,
    JaxOfflineController,
    load_config
)


env = pyRDDLGym.make('HVAC_ippc2023', '1')
if not os.path.exists('frames'):
    os.makedirs('frames')
env.horizon = 90   # just to speed things up
recorder = MovieGenerator("frames", "hvac", max_frames=env.horizon)
env.set_visualizer(viz=None, movie_gen=recorder)

agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
agent.evaluate(env, episodes=10, render=True)
env.close()

if not os.path.exists('frames'):
    os.makedirs('frames')
env.horizon = 30   # just to speed things up
recorder = MovieGenerator("frames", "HVAC", max_frames=env.horizon)
env.set_visualizer(viz=None, movie_gen=recorder)

#Image(filename='frames/traffic_0.gif') 
#########
import shutil
from pathlib import Path

src = Path("frames/HVAC_0.gif")
dst = Path.home() / "הורדות" / "HVAC_0.gif"

shutil.copy(src, dst)
print(f"Saved GIF to {dst}")