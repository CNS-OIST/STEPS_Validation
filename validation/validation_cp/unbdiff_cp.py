########################################################################

# 3D diffusion in an infinite volume from a point source. 
# CHECKPOINT

# AIMS: to verify checkpointing and restoring of the spatial stochastic 
# solver 'Tetexact' in the context of the Unbounded Diffusion model 
# (see validation/unbdiff.py)
  
########################################################################

from __future__ import print_function, absolute_import

import datetime
import steps.model as smodel
import numpy as np
import steps.solver as solvmod
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng
import time

from . import tol_funcs
from .. import configuration

rng = srng.create('mt19937', 1024) 
rng.initialize(int(time.time()%4294967295)) # The max unsigned long

# Number of iterations; plotting dt; sim endtime:
NITER = 10

DT = 0.01
INT = 0.21

# Number of molecules injected in centre; diff constant; number of tets sampled:
NINJECT = 100000 	

DCST = 0.02e-9

# With good code <1% fail with a tolerance of 5% 
tolerance = 5.0/100

########################################################################

SAMPLE = 32552	 # all tets

MESHFILE = 'sphere_rad10_33Ktets_adaptive'

# create the array of tet indices to be found at random
tetidxs = np.zeros(SAMPLE, dtype = 'int')
for i in range(SAMPLE): tetidxs[i] = i

# further create the array of tet barycentre distance to centre
tetrads = np.zeros(SAMPLE)
tetvols = np.zeros(SAMPLE)

########################################################################

def gen_model():
    
    mdl = smodel.Model()
    X = smodel.Spec('X', mdl)
    
    cytosolv = smodel.Volsys('cytosolv', mdl)
    dif_X = smodel.Diff('diffX', cytosolv, X)
    dif_X.setDcst(DCST)
    
    return mdl

########################################################################

def gen_geom():
    mesh = smeshio.loadMesh(configuration.mesh_path(MESHFILE))[0]
    ctetidx = mesh.findTetByPoint([0.0, 0.0, 0.0])
    
    ntets = mesh.countTets()
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    comp.addVolsys('cytosolv')
    
    # Now find the distance of the centre of the tets to the centre of the centre tet (at 0,0,0)
    cbaryc = mesh.getTetBarycenter(ctetidx)
    for i in range(SAMPLE):
        baryc = mesh.getTetBarycenter(int(tetidxs[i]))
        r2 = np.power((baryc[0]-cbaryc[0]),2) + np.power((baryc[1]-cbaryc[1]),2) + np.power((baryc[2]-cbaryc[2]),2)
        r = np.sqrt(r2)
        # Conver to microns
        tetrads[i] = r*1.0e6
        tetvols[i] = mesh.getTetVol(int(tetidxs[i]))
    
    return mesh

########################################################################

m = gen_model()
g = gen_geom()

# Fetch the index of the centre tet
ctetidx = g.findTetByPoint([0.0, 0.0, 0.0])
# And fetch the total number of tets to make the data structures
ntets = g.countTets()

sim = solvmod.Tetexact(m, g, rng)

tpnts = np.arange(0.0, INT, DT)
ntpnts = tpnts.shape[0]

#Create the big old data structure: iterations x time points x concentrations
res = np.zeros((NITER, ntpnts, SAMPLE))

sim.reset()
sim.setTetSpecCount(ctetidx, 'X', NINJECT)
sim.checkpoint(configuration.checkpoint('unbdiff'))

