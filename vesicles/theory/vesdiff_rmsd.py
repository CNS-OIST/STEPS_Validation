from matplotlib import pyplot as plt
import numpy as np
from math import *


root_mean_squared_disps=np.arange(0.02, 2.02, 0.02) #relative to diameter
proportion = []
for rmsd in root_mean_squared_disps:
    print ("RMSD =", rmsd, "diameters")
    n = 1000000
    dist=np.zeros(n)
    for i in range(n):
        dx = np.random.normal()*rmsd/sqrt(3)
        dy = np.random.normal()*rmsd/sqrt(3)
        dz = np.random.normal()*rmsd/sqrt(3)

        dist[i] = sqrt(dx**2+dy**2+dz**2)
    
    total_above_2diam = (dist>2.0).sum()
    
    proportion.append(total_above_2diam/n)


plt.plot(root_mean_squared_disps, np.array(proportion)*100)
fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)
plt.xlabel("RMSD (/diameter)")
plt.ylabel("Diffusion steps over 2 diameters (%)")
plt.savefig('vesiclediff_rmsd.pdf', dpi=300, bbox_inches='tight')

