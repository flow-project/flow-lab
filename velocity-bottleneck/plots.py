import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = "~/flow-lab/velocity-bottleneck/baseline_results/"
# specify csv files to read from
file_data_each = pd.read_csv(results_dir + "ramp_off_each1.csv")
file_data_mean = pd.read_csv(results_dir + "ramp_off_mean1.csv")

# columns
flow_in_mean = file_data_mean["FLOW_IN"]
flow_out_mean = file_data_mean["MEAN_FLOW_OUT"]

# group by flow min, then get the max or min
Max = file_data_each.groupby("FLOW_IN").max()
Min = file_data_each.groupby("FLOW_IN").min()
# group by flow, then get the sd
std = file_data_each.groupby("FLOW_IN").std()
one_std_up = np.array(flow_out_mean) + np.array(std["ALL_FLOW_OUT"])
one_std_down = np.array(flow_out_mean) - np.array(std["ALL_FLOW_OUT"])

# mean flow plot
fig, plots = plt.subplots(3, sharex="all", gridspec_kw={'hspace': 0.3})
plots[0].plot(flow_in_mean, flow_out_mean, 'b-')
plots[0].set_xlabel('Inflow (vehs/hour)', fontsize=6)
plots[0].set_ylabel('Outflow (vehs/hour)', fontsize=6)
plots[0].set_title("Mean Flows - Ramp Meter Off", fontsize=9)
plots[0].fill_between(flow_in_mean, Min["ALL_FLOW_OUT"], Max["ALL_FLOW_OUT"], alpha=0.15, color="b")
plots[0].fill_between(flow_in_mean, one_std_down, one_std_up, alpha=0.3, color="b")
plots[2].plot(flow_in_mean, flow_out_mean, 'b-')
plots[2].set_xlabel('Inflow (vehs/hour)', fontsize=6)
plots[2].set_ylabel('Outflow (vehs/hour)', fontsize=6)
plots[2].set_title("Ramp Meter Off and Ramp Meter On", fontsize=9)
# plots[2].fill_between(flow_in_mean, Min["ALL_FLOW_OUT"], Max["ALL_FLOW_OUT"], alpha=0.15, color="b")
# plots[2].fill_between(flow_in_mean, one_std_down, one_std_up, alpha=0.3, color="b")

# specify csv files to read from
file_data_each = pd.read_csv(results_dir + "ramp_on_each1.csv")
file_data_mean = pd.read_csv(results_dir + "ramp_on_mean1.csv")

# columns
flow_in_mean = file_data_mean["FLOW_IN"]
flow_out_mean = file_data_mean["MEAN_FLOW_OUT"]

# group by flow min, then get the max or min
Max = file_data_each.groupby("FLOW_IN").max()
Min = file_data_each.groupby("FLOW_IN").min()
# group by flow, then get the sd
std = file_data_each.groupby("FLOW_IN").std()
one_std_up = np.array(flow_out_mean) + np.array(std["ALL_FLOW_OUT"])
one_std_down = np.array(flow_out_mean) - np.array(std["ALL_FLOW_OUT"])

# mean flow plot
# fig, plots = plt.subplots(2)
plots[1].plot(flow_in_mean, flow_out_mean, 'r-')
plots[1].set_xlabel('Inflow (vehs/hour)', fontsize=6)
plots[1].set_ylabel('Outflow (vehs/hour)', fontsize=6)
plots[1].set_title("Mean Flows - Ramp Meter On", fontsize=9)
plots[1].fill_between(flow_in_mean, Min["ALL_FLOW_OUT"], Max["ALL_FLOW_OUT"], alpha=0.15, color="r")
plots[1].fill_between(flow_in_mean, one_std_down, one_std_up, alpha=0.3, color="r")
plots[2].plot(flow_in_mean, flow_out_mean, 'r-')
# plots[2].fill_between(flow_in_mean, Min["ALL_FLOW_OUT"], Max["ALL_FLOW_OUT"], alpha=0.15, color="r")
# plots[2].fill_between(flow_in_mean, one_std_down, one_std_up, alpha=0.3, color="r")
plt.show()
