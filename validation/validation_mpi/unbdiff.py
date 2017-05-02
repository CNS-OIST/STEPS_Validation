# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# STEPS - STochastic Engine for Pathway Simulation
# Copyright (C) 2007-2013 Okinawa Institute of Science and Technology, Japan.
# Copyright (C) 2003-2006 University of Antwerp, Belgium.
#
# See the file AUTHORS for details.
#
# This file is part of STEPS.
#
# STEPS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# STEPS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import datetime
import steps
import steps.mpi
import steps.model as smodel
import numpy as np
import steps.mpi.solver as solvmod
import steps.utilities.meshio as smeshio
import steps.utilities.geom_decompose as gd
import steps.geom as stetmesh
import steps.rng as srng
import time

from tol_funcs import *

rng = srng.create('r123', 1024)
rng.initialize(steps.mpi.rank + 1000)

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
    dif_X = smodel.Diff('diffX', cytosolv, X, 0)
    
    return mdl

########################################################################

def gen_geom():
    mesh = smeshio.loadMesh('meshes/'+MESHFILE)[0]
    
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

tet_hosts = gd.binTetsByAxis(g, steps.mpi.nhosts)

if steps.mpi.rank ==0:
    gd.printPartitionStat(tet_hosts)

sim = solvmod.TetOpSplit(m, g, rng, False, tet_hosts)

tpnts = np.arange(0.0, INT, DT)
ntpnts = tpnts.shape[0]

#Create the big old data structure: iterations x time points x concentrations
if steps.mpi.rank ==0:
    res = np.zeros((NITER, ntpnts, SAMPLE))

for j in range(NITER):
    if steps.mpi.rank == 0: print "iteration: ", j , "/", NITER
    sim.reset()
    sim.setTetCount(ctetidx, 'X', NINJECT)
    sim.setCompDiffD('cyto', 'diffX', DCST)
    for i in range(ntpnts):
        #if steps.mpi.rank == 0: print "tpnt: ", tpnts[i]
        sim.run(tpnts[i])
        for k in range(SAMPLE):
            count = sim.getTetCount(int(tetidxs[k]), 'X')
            if steps.mpi.rank ==0:
                res[j, i, k] = count
#print '%d / %d' % (j + 1, NITER)
ndiff = sim.getDiffExtent()
niteration = sim.getNIteration()
nreac = sim.getReacExtent()

if steps.mpi.rank == 0:
    itermeans = np.mean(res, axis = 0)

    tpnt_compare = [10, 15, 20]
    passed = False
    max_err = 0.0
    force_end = False
    for t in tpnt_compare:
        if force_end: break
        bin_n = 20
        
        r_max = tetrads.max()
        r_min = 0.0
        
        r_seg = (r_max-r_min)/bin_n
        bin_mins = np.zeros(bin_n+1)
        r_tets_binned = np.zeros(bin_n)
        bin_vols = np.zeros(bin_n)
        
        r = r_min
        for b in range(bin_n + 1):
            bin_mins[b] = r
            if (b!=bin_n): r_tets_binned[b] = r +r_seg/2.0
            r+=r_seg
        bin_counts = [None]*bin_n
        for i in range(bin_n): bin_counts[i] = []
        for i in range((itermeans[t].size)):
            i_r = tetrads[i]
            for b in range(bin_n):
                if(i_r>=bin_mins[b] and i_r<bin_mins[b+1]):
                    bin_counts[b].append(itermeans[t][i])
                    bin_vols[b]+=sim.getTetVol(int(tetidxs[i]))
                    break
        
        bin_concs = np.zeros(bin_n)
        for c in range(bin_n):
            for d in range(bin_counts[c].__len__()):
                bin_concs[c] += bin_counts[c][d]
            bin_concs[c]/=(bin_vols[c]*1.0e18)
        
        for i in range(bin_n):
            if (r_tets_binned[i] > 2.0 and r_tets_binned[i] < 6.0):
                rad = r_tets_binned[i]*1.0e-6
                det_conc = 1e-18*((NINJECT/(np.power((4*np.pi*DCST*tpnts[t]),1.5)))*(np.exp((-1.0*(rad*rad))/(4*DCST*tpnts[t]))))
                steps_conc = bin_concs[i]
                if tolerable(det_conc, steps_conc, tolerance):
                    passed = True
                else:
                    passed = False
                    force_end = True
                    break
                if (abs(2*(det_conc-steps_conc)/(det_conc+steps_conc)) > max_err): max_err = abs(2*(det_conc-steps_conc)/(det_conc+steps_conc))
    #print max_err*100.0, "%"
    print "max error", max_err*100.0, "%"
