
import numpy as np
from math import *

# the number of samples of the random steps
n = 1000000

root_mean_squared_disps=np.arange(0.02, 2.02, 0.04) #relative to diameter
proportion = []
for rmsd in root_mean_squared_disps:
    print ("RMSD =", rmsd, "diameters")
    dist=np.zeros(n)
    for i in range(n):
        dx = np.random.normal()*rmsd/sqrt(3)
        dy = np.random.normal()*rmsd/sqrt(3)
        dz = np.random.normal()*rmsd/sqrt(3)

        dist[i] = sqrt(dx**2+dy**2+dz**2)
    
    total_above_2diam = (dist>2.0).sum()
    
    proportion.append(total_above_2diam/n)

with open(f'data/rmsd_{n}.npy', 'wb') as f:
    np.save(f, root_mean_squared_disps)
    np.save(f, proportion)

