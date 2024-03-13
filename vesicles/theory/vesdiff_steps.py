import numpy as np
from random import *
from math import *


D=0.06e-12
dts = [1e-4, 1e-3]

# the number of samples of the random steps
n = 5000000

f = open('data/steps_'+str(n)+'.npy', 'wb')
np.save(f, dts)

for dt in dts:
    dist=np.zeros(n)
    
    for i in range(n):
        dx = np.random.normal()*sqrt(2*D*dt)
        dy = np.random.normal()*sqrt(2*D*dt)
        dz = np.random.normal()*sqrt(2*D*dt)

        dist[i] = sqrt(dx**2+dy**2+dz**2)*1e9
    
    np.save(f, dist)
