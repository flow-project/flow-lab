import pandas as pd
import matplotlib.pyplot as plt

results_dir = "~/flow-lab/velocity-bottleneck/baseline_results/"

file_data_each = pd.read_csv(results_dir + "ramp_off_each.csv")
file_data_mean = pd.read_csv(results_dir + "ramp_off_mean.csv")

#columns
flow_in_each = file_data_each["FLOW_IN"]
flow_out_each = file_data_each["ALL_FLOW_OUT"]
flow_in_mean = file_data_mean["FLOW_IN"]
flow_out_mean = file_data_mean["MEAN_FLOW_OUT"]

#mean flow plot
fig, plots = plt.subplots(2)
plots[0].plot(flow_in_mean, flow_out_mean, 'b-')
plots[0].set(xlabel='Inflow (vehs/hour)', ylabel='Outflow (vehs/hour)')
plots[0].set_title("Mean Flows")

#each flow plot
plots[1].scatter(flow_in_each, flow_out_each)
plots[1].set(xlabel='Inflow (vehs/hour)', ylabel='Outflow (vehs/hour)')
plots[1].set_title("Each Flows")
plt.show()

#TODO: Plot both plots on one

# #RAMP METER PLOTS
# file_data_each_rm = pd.read_csv(results_dir + "ramp_on_each.csv")
# file_data_mean_rm = pd.read_csv(results_dir + "ramp_on_mean.csv")
#
# #columns
# flow_in_each_rm = file_data_each_rm["FLOW_IN"]
# flow_out_each_rm = file_data_each_rm["ALL_FLOW_OUT"]
# flow_in_mean_rm = file_data_mean_rm["FLOW_IN"]
# flow_out_mean_rm = file_data_mean_rm["MEAN_FLOW_OUT"]
#
# #mean flow plot
# fig, plots = plt.subplots(2)
# plots[0].plot(flow_in_mean_rm, flow_out_mean_rm, 'b-')
# plots[0].set(xlabel='Inflow (vehs/hour)', ylabel='Outflow (vehs/hour)')
# plots[0].set_title("Mean Flows (RAMP METER ON)")
#
# #each flow plot
# plots[1].scatter(flow_in_each_rm, flow_out_each_rm)
# plots[1].set(xlabel='Inflow (vehs/hour)', ylabel='Outflow (vehs/hour)')
# plots[1].set_title("Each Flows (RAMP METER ON)")
# plt.show()