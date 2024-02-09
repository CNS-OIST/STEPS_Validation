from matplotlib import pyplot as plt
import numpy as np
from random import *
from math import *


D=0.06e-12

for t in [1e-4, 1e-3]:
    n = 5000000
    dist=np.zeros(n)
    
    for i in range(n):

        dx = np.random.normal()*sqrt(2*D*t) 
        dy = np.random.normal()*sqrt(2*D*t)
        dz = np.random.normal()*sqrt(2*D*t)

        dist[i] = sqrt(dx**2+dy**2+dz**2)*1e9
    
    plt.hist(dist, bins=np.arange(0, 50, 0.5), density=True, label='${\Delta t}$='+str(t*1e3)+' ms', alpha=0.7)

fig = plt.gcf()
fig.set_size_inches(3.4, 3.4)

plt.xlabel("Vesicle diffusion distance (nm)")
plt.ylabel("Probability density")
plt.legend()
plt.savefig('plots/vesiclediff_steps.pdf', dpi=300, bbox_inches='tight')

