import steps.interface
from steps.saving import *


import pickle
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


with HDF5Handler('data/rothman') as hdf:
    group = hdf['rothman']
    positions, = group.results
    locals().update(group.parameters)
    tpnts = np.array(group.staticData['tpnts'])

    ves_mean = np.mean(np.sum(
        (positions.data[0, ...] - starting_pos)**2, axis=2) * 1e12,
        axis=1)

    plt.plot(tpnts, ves_mean[1:] / (tpnts * 6), linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('D ($\mu$$m^2$/s)')
    plt.ylim(0, 0.06)
    fig = plt.gcf()
    fig.set_size_inches(3.4, 3.4)
    fig.savefig("plots/rothman.pdf", dpi=300, bbox_inches='tight')
    plt.close()
