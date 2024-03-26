import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


# Analytical, single exponentia
dwelltime_single_anl = []
for i in range(100000): dwelltime_single_anl.append(random.expovariate(1/0.045))

plt.subplot(221)
plt.hist (dwelltime_single_anl, bins=np.arange(0,0.5,0.015), label='1-step process 0.045s mean', density=True)
plt.xlim(0,0.3)
plt.legend()


# STEPS sim, single exponential
with open('data/path_dwelltimes_single.npy', 'rb') as f: dwelltime_single = np.load(f)
plt.subplot(222)
plt.hist (dwelltime_single, bins=np.arange(0,0.5,0.015), label='STEPS simulation 1-step process', density=True)
plt.xlim(0,0.3)
plt.legend()


# Analytical, double exponential, ratio 0.7:0.3
dwelltime_double_anl = []
for i in range(100000): dwelltime_double_anl.append(random.expovariate((1/0.3)/0.045)+random.expovariate((1/0.7)/0.045))
plt.subplot(223)
plt.hist(dwelltime_double_anl, bins=np.arange(0,0.5,0.015), label='2-step 0.045s mean, 0.7:0.3', density=True)
plt.xlim(0,0.3)
plt.xlabel('Dwelltime (s)')
plt.legend()


# STEPS sim, double exponential, ratio 0.7:0.3
with open('data/path_dwelltimes_double.npy', 'rb') as f: dwelltime_double = np.load(f)
plt.subplot(224)
plt.hist(dwelltime_double, bins=np.arange(0,0.5,0.015), label='STEPS simulation 2-step, 0.7:0.3', density=True)
plt.xlim(0,0.3)
plt.xlabel('Dwelltime (s)')
plt.legend()


fig = plt.gcf()
fig.set_size_inches(9, 9)
plt.subplots_adjust(wspace=0.3, hspace=None)
fig.savefig("plots/path_dwelltimes.pdf", dpi=300, bbox_inches='tight')
