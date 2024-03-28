import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# the number of samples of the random steps
n = 5000000

with open(f'data/steps_{n}.npy', 'rb') as f:
    dts = np.load(f)

    for dt in dts:
        dist = np.load(f)
        plt.hist(dist, bins=np.arange(0, 50, 0.5), density=True, label='${\Delta t}$='+str(dt*1e3)+' ms', alpha=0.7)

fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)

plt.xlabel("Vesicle diffusion distance (nm)")
plt.ylabel("Probability density")
plt.legend()
plt.savefig('plots/vesiclediff_steps.pdf', dpi=300, bbox_inches='tight')

