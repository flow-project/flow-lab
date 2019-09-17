import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

results_dir = "emission"
edges_of_interest = ["left","center"]

fig, ax = plt.subplots(dpi=300)
for i in [0.025, 0.05, 0.1]:
    csvs = glob.glob(os.path.expanduser(
        results_dir)+"/*_{:.3f}_*".format(i))
    if len(csvs) > 0:
        data = pd.read_csv(csvs[-1])
        
        cars_in_highway = data[data.edge_id.isin(edges_of_interest)]
        cars_in_highway['x'] -= 100
        num_cars = cars_in_highway.groupby("time")['id'].nunique()
        ax.plot(num_cars[::5], label="{:.1f}% RL veh".format(i*100))
ax.legend()
ax.set_xlabel("time (s)")
ax.set_ylabel("number of vehicles")
fig.savefig("num_vehicles.pdf", bbox_inches="tight")