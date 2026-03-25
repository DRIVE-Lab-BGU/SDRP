import csv
import sys

import matplotlib as mpl
mpl.use("MacOSX")  # on macOS; or "TkAgg" / "QtAgg"
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

header = ['epoch', 'eval_reward', 'eval_std']
# file_names = ['reservoir_instance3_deterministic_1000.csv', 'reservoir_instance3_stochastic1_1000.csv', 'reservoir_instance3_stochastic3_1000.csv']
# file_names = ['reservoir_instance4_deterministic_250.csv', 'reservoir_instance4_stochastic1_250.csv', 'reservoir_instance4_stochastic3_250.csv']
# file_names = ['reservoir_instance4_stochastic0_500.csv', 'reservoir_instance4_stochastic1_500.csv', 'reservoir_instance4_stochastic3_500.csv']
# file_names = ['reservoir_instance5_stochastic0_500.csv', 'reservoir_instance5_stochastic1_500.csv', 'reservoir_instance5_stochastic3_500.csv']
# file_names = ['cartpole_instance1_stochastic0_200.csv','cartpole_instance1_stochastic1_200.csv']
# file_names = ['reservoir_instance4_stochastic0_500.csv', 'reservoir_instance3_noise3_1000.csv']
# file_names = ['HVAC_instance5_noise0_100.csv', 'HVAC_instance5_noise3_100.csv']
# file_names = ['UAV_instance2_noise0_100.csv', 'UAV_instance2_noise3_100.csv']

# file_names = ['HVAC_instance1_noise0.csv', 'HVAC_instance1_noise1.csv']#, 'HVAC_instance1_noise0_IS.csv','HVAC_instance1_noise3.csv', 'HVAC_instance1_noise3_IS.csv']#, 'HVAC_instance1_noise3.csv']
# legend = ['Backprop', 'noise1']#, 'noise1_IS', 'noise3', 'noise3_IS']

file_names = ['HVAC_instance1_noise0_IS.csv', 'HVAC_instance1_noise5_IS.csv', 'HVAC_instance1_noise7_IS.csv']
legend = ['no noise', 'noise5', 'noise7']

# file_names = ['HVAC_instance1_noise0.csv']
# legend = ['Backprop']
# file_names = ['reservoir_instance3_noise0_1000.csv', 'reservoir_instance3_noise3_1000.csv']
# file_names = ['reservoir_instance5_stochastic0_500.csv', 'reservoir_instance5_noise3_IS.csv', 'reservoir_instance5_noise5_IS.csv', 'reservoir_instance5_noise7_IS.csv']
# file_names = ['reservoir_instance3_noise3_IS.csv', 'reservoir_instance3_noise3_1000.csv']
# file_names = ['reservoir_instance3_noise0_1000.csv', 'reservoir_instance3_noise3_1000.csv', 'reservoir_instance3_noise3_IS.csv', 'reservoir_instance3_noise5_IS.csv','reservoir_instance3_noise5_noIS.csv']
# file_names = ['reservoir_instance5_stochastic0_500.csv','reservoir_instance5_noise3.csv','reservoir_instance5_noise5.csv','reservoir_instance5_noise7.csv',
#               'reservoir_instance5_noise3_IS.csv','reservoir_instance5_noise5_IS.csv','reservoir_instance5_noise7_IS.csv']
# legend = ['no noise','noise3', 'noise5', 'noise7']
# legend = ['no noise','noise3', 'noise5','noise7','noise3IS', 'noise5IS','noise7IS']
# legend = ['w/o exploration noise', 'noise=3, w/o IS', 'noise=3, w IS', 'noise=5, w IS', 'noise=5, w/o IS']
title = "Multi-zone HVAC Control - 2 zones, 3 heaters"
# legend = []
base_path = os.path.dirname(os.path.abspath(__file__))

files = []
for file in file_names:
  files.append(os.path.join(base_path, 'logs', file))


if not files:
  raise ValueError("No files provided.")

# Normalize legend length
if legend is None or len(legend) != len(files):
  legend = [Path(f).stem for f in files]  # fallback to filename stems

fig, ax = plt.subplots(figsize=(9, 5))
line_handles = []

for fpath, label in zip(files, legend):
  epochs, vals, stds = [], [], []

  # Read with csv.DictReader
  with open(fpath, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    # Ensure required headers are present
    required = {"epoch", "eval_reward", "eval_std"}
    if not required.issubset(reader.fieldnames):
      raise ValueError(f"{fpath} is missing required columns: {required - set(reader.fieldnames)}")

    for row in reader:
      try:
        epochs.append(int(row["epoch"]))
        vals.append(float(row["eval_reward"]))
        stds.append(float(row["eval_std"]))
      except ValueError:
        # Skip rows with invalid numeric values
        continue

  # Sort by epoch
  data = sorted(zip(epochs, vals, stds), key=lambda x: x[0])
  epochs, vals, stds = zip(*data)
  cutoff = 100

  # Plot line with blue squares
  line, = ax.plot(epochs[:cutoff], vals[:cutoff], marker="s", linestyle="-")
  line_handles.append(line)

  # Std envelope (± std)
  upper = [v + e for v, e in zip(vals, stds)]
  lower = [v - e for v, e in zip(vals, stds)]
  ax.fill_between(epochs[:cutoff], lower[:cutoff], upper[:cutoff], alpha=0.3, label="_nolegend_")

random = -1119993.4009455156*np.ones([10000,1])
# line, = ax.plot(epochs[:cutoff], random[:cutoff], marker="s", linestyle="-")
# ax.fill_between(epochs[:cutoff], lower[:cutoff], upper[:cutoff], alpha=0.3, label="_nolegend_")
# line_handles.append(line)
# legend.append('random')

ax.set_xlabel("Episode", fontsize=16)
ax.set_ylabel("Eval reward", fontsize=16)
ax.set_title(title)
ax.grid(True, linestyle="--", alpha=0.6)
fig.tight_layout()

# ✅ Build legend from the line handles only
ax.legend(line_handles, legend, handlelength=2.5, frameon=True, fontsize=12)

plt.show()

