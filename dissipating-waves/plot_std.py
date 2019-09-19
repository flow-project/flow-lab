import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import numpy as np

# some plotting parameters
cdict = {
    'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
    'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
    'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
}
my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)


def make_segments(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

results_dir = "./emission"
edges_of_interest = ["inflow_highway","left","center", ':center_0', ':center_1']

fig, ax = plt.subplots(dpi=300)
penetration = 0.05
csvs = glob.glob(os.path.expanduser(
    results_dir)+"/*_{:.3f}_*".format(penetration))
time_horizon = 400
if len(csvs) > 0:
    data = pd.read_csv(csvs[-1])
    cars_in_highway = data[data.edge_id.isin(edges_of_interest)]
    # eliminate transient
    cars_in_highway['time'] -= 400
    cars_in_highway = cars_in_highway[(cars_in_highway['time']<=time_horizon) & (cars_in_highway['time']>=0)]
    norm = plt.Normalize(0.0, cars_in_highway.speed.max())
    fig, ax = plt.subplots(dpi=300)
    for veh, trajectory in cars_in_highway[['time', 'id', 'x', 'speed']].groupby('id'):
        segments = make_segments(trajectory.time, trajectory.x)
        lc = LineCollection(segments, array=trajectory.speed.values[1:], cmap=my_cmap,
                            norm=norm,
                            linewidth=1, alpha=1)
        ax.add_collection(lc)

    fig.colorbar(lc, ax=ax)
    ax.set_xlim(right=time_horizon)
    ax.set_ylim(top=700)
    fig.savefig("std.png", dpi=600, bbox_inches="tight")