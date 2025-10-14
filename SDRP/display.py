import csv
import sys

import matplotlib as mpl
mpl.use("MacOSX")  # on macOS; or "TkAgg" / "QtAgg"
import matplotlib.pyplot as plt
from pathlib import Path
import os

header = ['epoch', 'eval_reward', 'eval_std']
file_names = ['reservoir_instance3_deterministic_1000.csv', 'reservoir_instance3_stochastic1_1000.csv', 'reservoir_instance3_stochastic3_1000.csv']
# file_names = ['reservoir_instance4_deterministic_250.csv', 'reservoir_instance4_stochastic1_250.csv', 'reservoir_instance4_stochastic3_250.csv']
file_names = ['reservoir_instance4_stochastic0_500.csv', 'reservoir_instance4_stochastic1_500.csv', 'reservoir_instance4_stochastic3_500.csv']
file_names = ['reservoir_instance5_stochastic0_500.csv', 'reservoir_instance5_stochastic1_500.csv', 'reservoir_instance5_stochastic3_500.csv']
legend = ['deterministic', 'stochastic_1', 'stochastic_3']

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
# plt.title(title)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()






# # plots = len(files)
# plots = {}
# for file in files:
#   with open(os.path.join(base_path, 'logs', file), mode='r', newline='') as f:
#     csv_reader = csv.reader(file)
#     epoch = []
#     reward = []
#     std = []
#     for row in csv_reader:
#       if row == header:
#         continue
#       epoch.append(row[0])
#       reward.append(row[1])
#   plots[file] = { 'epochs': epoch, 'reward': reward, 'std': std }
#
# sys.exit()


#deterministic
# with open(os.path.join(base_path, 'logs', deterministic_csv), mode='r', newline='') as file:
#   csv_reader = csv.reader(file)
#   d_x_axis = []
#   d_y_axis = []
#   for row in csv_reader:
#     if row == header:
#       continue
#     d_x_axis.append(row[0])
#     d_y_axis.append(row[1])
#
# #stochastic
# with open(os.path.join(base_path, 'logs', stochastic1_csv), mode='r', newline='') as file:
#   csv_reader = csv.reader(file)
#   s_x_axis = []
#   s_y1_axis = []
#   for row in csv_reader:
#     if row == header:
#       continue
#     s_x_axis.append(row[0])
#     s_y1_axis.append(row[1])
#
# with open(os.path.join(base_path, 'logs', stochastic2_csv), mode='r', newline='') as file:
#   csv_reader = csv.reader(file)
#   s_x_axis = []
#   s_y2_axis = []
#   for row in csv_reader:
#     if row == header:
#       continue
#     s_x_axis.append(row[0])
#     s_y2_axis.append(row[1])


# x_axis = np.array(d_x_axis)
# d_y_axis = np.array(d_y_axis, dtype=float)
# s_y1_axis = np.array(s_y1_axis, dtype=float)
# s_y2_axis = np.array(s_y2_axis, dtype=float)
# plt.figure(figsize=(10, 6)) # Adjust figure size as needed
# plt.plot(x_axis, d_y_axis, marker='o', linestyle='-') # Line plot with markers
# plt.plot(x_axis, s_y1_axis, marker='x', linestyle='-') # Line plot with markers
# plt.plot(x_axis, s_y2_axis, marker='v', linestyle='-') # Line plot with markers
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')
# plt.title('Plot Title')
# plt.legend(['Deterministic', 'Stochastic_1', 'Stochastic_3'])
# plt.grid(True) # Add a grid for better readability
# plt.show()