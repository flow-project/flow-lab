import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import csv
from collections import defaultdict

results_dir = "~/ray_results/dissipating_waves/"

columns = ['episode_reward_mean', 'episode_reward_min',
           'episode_reward_max', 'training_iteration']
res_dirs = {}
for i in [0.025, 0.05, 0.1]:
    folders = glob.glob(os.path.expanduser(
        results_dir)+"/*_{:.3f}_*/".format(i))
    if len(folders) > 0:
        # Assuming for now that these folders are unique
        filepath = _[0] + "progress.csv"
        rewards_data = defaultdict(list)
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    rewards_data[col].append(float(row[col]))

        plt.plot(rewards_data['training_iteration'],
                 rewards_data['episode_reward_mean'], label="{:.1f}% RL veh".format(i*100))
        plt.fill_between(rewards_data['training_iteration'],
                         rewards_data['episode_reward_min'], rewards_data['episode_reward_max'], alpha=0.5)

plt.legend()
plt.xlabel("iterations")
plt.ylabel("mean reward")
plt.savefig("results.pdf", bbox_inches="tight")
