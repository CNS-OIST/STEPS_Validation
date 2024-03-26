import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# the number of samples of the random steps
n = 1000000

with open('data/rmsd_'+str(n)+'.npy', 'rb') as f:
    root_mean_squared_disps = np.load(f)
    proportion = np.load(f)

plt.plot(root_mean_squared_disps, np.array(proportion)*100, marker='o', ms=3)
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
plt.xlabel("RMSD (/diameter)")
plt.ylabel("Diffusion steps over 2 diameters (%)")
plt.savefig('plots/vesiclediff_rmsd.pdf', dpi=300, bbox_inches='tight')

