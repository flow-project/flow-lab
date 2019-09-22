import csv
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "~/flow-lab/velocity_bottleneck/baseline_results/"

file_data = pd.read_csv(results_dir + "ramp_off.csv")
flow_in = file_data[0]
flow_out = flow_in[1]

plt.plot(flow_in, flow_out)
plt.show()