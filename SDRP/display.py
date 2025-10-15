import csv
import sys

import matplotlib as mpl
mpl.use("MacOSX")  # on macOS; or "TkAgg" / "QtAgg"
import matplotlib.pyplot as plt
from pathlib import Path
import os

header = ['epoch', 'eval_reward', 'eval_std']
# file_names = ['reservoir_instance3_deterministic_1000.csv', 'reservoir_instance3_stochastic1_1000.csv', 'reservoir_instance3_stochastic3_1000.csv']
# file_names = ['reservoir_instance4_deterministic_250.csv', 'reservoir_instance4_stochastic1_250.csv', 'reservoir_instance4_stochastic3_250.csv']
# file_names = ['reservoir_instance4_stochastic0_500.csv', 'reservoir_instance4_stochastic1_500.csv', 'reservoir_instance4_stochastic3_500.csv']
# file_names = ['reservoir_instance5_stochastic0_500.csv', 'reservoir_instance5_stochastic1_500.csv', 'reservoir_instance5_stochastic3_500.csv']
# file_names = ['cartpole_instance1_stochastic0_200.csv','cartpole_instance1_stochastic1_200.csv']
file_names = ['reservoir_instance3_noise0_1000.csv', 'reservoir_instance3_noise3_1000.csv']
file_names = ['HVAC_instance5_noise0_100.csv', 'HVAC_instance5_noise3_100.csv']
legend = ['deterministic', 'stochastic_3']

base_path = os.path.dirname(os.path.abspath(__file__))

files = []
for file in file_names:
  files.append(os.path.join(base_path, 'logs', file))


if not files:
  raise ValueError("No files provided.")

# Normalize legend length
if legend is None or len(legend) != len(files):
  legend = [Path(f).stem for f in files]  # fallback to filename stems

plt.figure(figsize=(9, 5))

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

  # Plot line with blue squares
  plt.plot(epochs, vals, marker="s", linestyle="-", label=label)

  # Std envelope (± std)
  upper = [v + e for v, e in zip(vals, stds)]
  lower = [v - e for v, e in zip(vals, stds)]
  plt.fill_between(epochs, lower, upper, alpha=0.3)

plt.xlabel("Epoch")
plt.ylabel("Eval reward")
plt.title('Reservoir domain, instance 3')
plt.legend(legend)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


