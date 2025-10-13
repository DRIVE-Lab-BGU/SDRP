import csv
import sys
import numpy as np

import matplotlib as mpl
mpl.use("MacOSX")  # on macOS; or "TkAgg" / "QtAgg"
import matplotlib.pyplot as plt
import os

header = ['epoch', 'eval_reward']
base_path = os.path.dirname(os.path.abspath(__file__))
deterministic_csv = 'reservoir_instance3_deterministic_1000.csv'
# deterministic_csv = 'reservoir_instance3_stochastic1_1000.csv'
stochastic1_csv = 'reservoir_instance3_stochastic1_1000.csv'
# stochastic2_csv = 'reservoir_instance3_deterministic_1000.csv'

#deterministic
with open(os.path.join(base_path, 'logs', deterministic_csv), mode='r', newline='') as file:
  csv_reader = csv.reader(file)
  d_x_axis = []
  d_y_axis = []
  for row in csv_reader:
    if row == header:
      continue
    d_x_axis.append(row[0])
    d_y_axis.append(row[1])

#stochastic
with open(os.path.join(base_path, 'logs', stochastic1_csv), mode='r', newline='') as file:
  csv_reader = csv.reader(file)
  s_x_axis = []
  s_y1_axis = []
  for row in csv_reader:
    if row == header:
      continue
    s_x_axis.append(row[0])
    s_y1_axis.append(row[1])
#
# with open(os.path.join(base_path, 'logs', stochastic1_csv), mode='r', newline='') as file:
#   csv_reader = csv.reader(file)
#   s_x_axis = []
#   s_y2_axis = []
#   for row in csv_reader:
#     if row == header:
#       continue
#     s_x_axis.append(row[0])
#     s_y2_axis.append(row[1])


x_axis = np.array(d_x_axis)
d_y_axis = np.array(d_y_axis, dtype=float)
s_y1_axis = np.array(s_y1_axis, dtype=float)
# s_y2_axis = np.array(s_y2_axis, dtype=float)
plt.figure(figsize=(10, 6)) # Adjust figure size as needed
plt.plot(x_axis, d_y_axis, marker='o', linestyle='-') # Line plot with markers
plt.plot(x_axis, s_y1_axis, marker='x', linestyle='-') # Line plot with markers
# plt.plot(x_axis, s_y2_axis, marker='v', linestyle='-') # Line plot with markers
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Plot Title')
plt.legend(['Deterministic', 'Stochastic_1'])
plt.grid(True) # Add a grid for better readability
plt.show()