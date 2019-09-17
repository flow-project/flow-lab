import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def make_segments(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

results_dir = "./emission"
edges_of_interest = ["left","center"]

fig, ax = plt.subplots(dpi=300)
penetration = 0.05
csvs = glob.glob(os.path.expanduser(
    results_dir)+"/*_{:.3f}_*".format(penetration))
if len(csvs) > 0:
    data = pd.read_csv(csvs[-1])
    cars_in_highway = data[data.edge_id.isin(edges_of_interest)]
    cars_in_highway['x'] -= 100

    norm = plt.Normalize(0.0, cars_in_highway.speed.max())
    fig, ax = plt.subplots(dpi=300)
    for veh, trajectory in cars_in_highway[['time', 'id', 'x', 'speed']].groupby('id'):
        segments = make_segments(trajectory.time, trajectory.x)
        lc = LineCollection(segments, array=trajectory.speed.values[1:], cmap='RdYlGn',
                            norm=norm,
                            linewidth=1, alpha=1)
        ax.add_collection(lc)

    fig.colorbar(lc, ax=ax)
    ax.set_xlim(right=360)
    ax.set_ylim(top=700)
    fig.savefig("std.pdf", bbox_inches="tight")