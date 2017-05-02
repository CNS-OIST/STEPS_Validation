########################################################################

# 2D diffusion on an infinite surface from point source. 

# AIMS: to verify STEPS spatial-stochastic solver 'Tetexact' supports 
# local initial conditions and calculates surface diffusion rates correctly.

# STEPS equivalent model: Stochastic 2D diffusion on one outer surface 
# of a disk; central triangle source.

# This model is equivalet to that described at
# http://www.biomedcentral.com/content/supplementary/1752-0509-6-36-s4.pdf
# "3D diffusion in infinite volume"
# with a 2D compartments ("atch") instead of a 3D compartments.

# Verification also takes place of model and mesh construction 
# components, particularly mesh loading and manipulation capabilities 
# with functions such as steps.geom.Tetmesh.getSurfTris and 
# steps.geom.Tetmesh.getTriArea etc. 
# Localised recording by steps.solver.Tetexact.getTriCount is also verified. 

# A 6% tolerance is imposed when comparing the mean output from 100 
# stochastic simulations of the STEPS model to the analytical solution. 
# There is an expected probability of failure of < 1%.

########################################################################
import steps
import steps.mpi
import steps.mpi.solver as solvmod
import steps.utilities.geom_decompose as gd

import steps.model as smodel
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng


import numpy as np
import time
import datetime

from tol_funcs import *


rng = srng.create('r123', 1024)
rng.initialize(steps.mpi.rank + 1000)

# Number of iterations; plotting dt; sim endtime:
NITER = 100

DT = 0.01
INT = 0.21

# Number of molecules injected in centre; diff constant; number of tets sampled:
NINJECT = 1000	

DCST = 0.02e-9

# With good code <1% fail with a tolerance of 6% 
tolerance = 6.0/100

########################################################################

MESHFILE = 'coin_10r_1h_13861.inp'

########################################################################

def gen_model():
    
    mdl = smodel.Model()
    X = smodel.Spec('X', mdl)
    
    ssys = smodel.Surfsys('ssys', mdl)
    dif_X = smodel.Diff('diffX', ssys, X)
    dif_X.setDcst(0)
    
    return mdl

########################################################################

def gen_geom():
    mesh = smeshio.loadMesh('meshes/'+MESHFILE)[0]
    
    ctetidx = mesh.findTetByPoint([0.0, 0.0, 0.5e-6])
    
    ntets = mesh.countTets()
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    
    alltris = mesh.getSurfTris()
    
    patch_tris = []
    for t in alltris:
        vert0, vert1, vert2 = mesh.getTri(t)
        if (mesh.getVertex(vert0)[2] > 0.0 and mesh.getVertex(vert1)[2] > 0.0 and mesh.getVertex(vert2)[2] > 0.0):
            patch_tris.append(t)
    
    patch_tris_n = len(patch_tris)
                
    patch = stetmesh.TmPatch('patch', mesh, patch_tris, icomp = comp)
    patch.addSurfsys('ssys')
    
    trirads = np.zeros(patch_tris_n)
    triareas = np.zeros(patch_tris_n)
                
    # TRy to find the central tri
    ctet_trineighbs = mesh.getTetTriNeighb(ctetidx)
    ctri_idx=-1
    for t in ctet_trineighbs: 
        if t in patch_tris:
            ctri_idx = t
    
    # Now find the distance of the centre of the tets to the centre of the centre tet (at 0,0,0)
    cbaryc = mesh.getTriBarycenter(ctri_idx)
    for i in range(patch_tris_n):
        baryc = mesh.getTriBarycenter(patch_tris[i])
        r2 = np.power((baryc[0]-cbaryc[0]),2) + np.power((baryc[1]-cbaryc[1]),2) + np.power((baryc[2]-cbaryc[2]),2)
        r = np.sqrt(r2)
        # Conver to microns
        trirads[i] = r*1.0e6
        triareas[i] = mesh.getTriArea(patch_tris[i])
    
    
    return mesh, patch_tris, patch_tris_n, ctri_idx, trirads, triareas

########################################################################

"Surface Diffusion - Unbounded, point source (Tetexact), mpi"

m = gen_model()
g, patch_tris, patch_tris_n, ctri_idx, trirads, triareas = gen_geom()
tet_hosts = gd.binTetsByAxis(g, steps.mpi.nhosts)
tri_hosts = gd.partitionTris(g, tet_hosts, patch_tris)

if steps.mpi.rank ==0:
    gd.printPartitionStat(tet_hosts, tri_hosts)

sim = solvmod.TetOpSplit(m, g, rng, False, tet_hosts, tri_hosts)

tpnts = np.arange(0.0, INT, DT)
ntpnts = tpnts.shape[0]

#Create the big old data structure: iterations x time points x concentrations
if steps.mpi.rank ==0:
    res_count = np.zeros((NITER, ntpnts, patch_tris_n))
    res_conc = np.zeros((NITER, ntpnts, patch_tris_n))

for j in range(NITER):
    sim.reset()
    for t in patch_tris:
        sim.setTriDiffD(t, 'diffX', DCST)
    sim.setTriCount(ctri_idx, 'X', NINJECT)
    for i in range(ntpnts):
        sim.run(tpnts[i])
        for k in range(patch_tris_n):
            count = sim.getTriCount(patch_tris[k], 'X')
            if steps.mpi.rank ==0:
                res_count[j, i, k] = count
                res_conc[j, i, k] = count/sim.getTriArea(patch_tris[k])

ndiff = sim.getDiffExtent()
niteration = sim.getNIteration()
nreac = sim.getReacExtent()


if steps.mpi.rank == 0:
    itermeans_count = np.mean(res_count, axis = 0)
    itermeans_conc = np.mean(res_conc, axis = 0)


    tpnt_compare = [12, 16, 20]
    passed = True
    max_err = 0.0
    
    force_end = False
    for t in tpnt_compare:
        if force_end: break
        bin_n = 20
        
        r_max = 0.0 	
        for i in trirads: 		
            if (i > r_max): r_max = i 	
        
        r_min = 0.0
        
        r_seg = (r_max-r_min)/bin_n
        bin_mins = np.zeros(bin_n+1)
        r_tris_binned = np.zeros(bin_n)
        bin_areas = np.zeros(bin_n)    
        
        r = r_min
        for b in range(bin_n + 1):
            bin_mins[b] = r
            if (b!=bin_n): r_tris_binned[b] = r +r_seg/2.0
            r+=r_seg
        bin_counts = [None]*bin_n
        for i in range(bin_n): bin_counts[i] = []
        for i in range((itermeans_count[t].size)):
            i_r = trirads[i]
            for b in range(bin_n):
                if(i_r>=bin_mins[b] and i_r<bin_mins[b+1]):
                    bin_counts[b].append(itermeans_count[t][i])
                    bin_areas[b]+=sim.getTriArea(int(patch_tris[i]))
                    break
        
        bin_concs = np.zeros(bin_n)
        for c in range(bin_n): 
            for d in range(bin_counts[c].__len__()):
                bin_concs[c] += bin_counts[c][d]
            bin_concs[c]/=(bin_areas[c]*1.0e12)
        
        for i in range(bin_n):
            if (r_tris_binned[i] > 2.0 and r_tris_binned[i] < 5.0):
                rad = r_tris_binned[i]*1.0e-6
                det_conc = 1.0e-12*(NINJECT/(4*np.pi*DCST*tpnts[t]))*(np.exp((-1.0*(rad*rad))/(4*DCST*tpnts[t])))
                steps_conc = bin_concs[i]
                if tolerable(det_conc, steps_conc, tolerance):
                    passed = True
                else:
                    passed = False
                    force_end = True
                    break
"""
                if (abs(2*(det_conc-steps_conc)/(det_conc+steps_conc)) > max_err): max_err = abs(2*(det_conc-steps_conc)/(det_conc+steps_conc))
    print "max error", max_err*100.0, "%"
"""

########################################################################
# END
