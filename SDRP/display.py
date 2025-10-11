import csv
import sys
import numpy as np

import matplotlib as mpl
mpl.use("MacOSX")  # on macOS; or "TkAgg" / "QtAgg"
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(os.path.abspath(__file__))
deterministic_csv = 'reservoir_instance3_deterministic_250.csv'
stochastic_csv = 'reservoir_instance3_stochastic1_250.csv'

#deterministic
with open(os.path.join(base_path, 'logs', deterministic_csv), mode='r', newline='') as file:
  csv_reader = csv.reader(file)
  d_x_axis = []
  d_y_axis = []
  for row in csv_reader:
    d_x_axis.append(row[0])
    d_y_axis.append(row[1])

#stochastic
with open(os.path.join(base_path, 'logs', stochastic_csv), mode='r', newline='') as file:
  csv_reader = csv.reader(file)
  s_x_axis = []
  s_y_axis = []
  for row in csv_reader:
    s_x_axis.append(row[0])
    s_y_axis.append(row[1])


x_axis = np.array(d_x_axis)
d_y_axis = np.array(d_y_axis, dtype=float)
s_y_axis = np.array(s_y_axis, dtype=float)
plt.figure(figsize=(10, 6)) # Adjust figure size as needed
plt.plot(x_axis, d_y_axis, marker='o', linestyle='-') # Line plot with markers
plt.plot(x_axis, s_y_axis, marker='x', linestyle='-') # Line plot with markers
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Plot Title')
plt.legend(['Deterministic', 'Stochastic'])
plt.grid(True) # Add a grid for better readability
plt.show()