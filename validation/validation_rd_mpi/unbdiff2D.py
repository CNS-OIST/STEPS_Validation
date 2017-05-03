########################################################################

# This is the parallel version of unbdiff2D.py in validation_rd.
# Use for parallel TetOpSplit validation.

########################################################################

import steps.model as smodel
import steps.mpi
import steps.mpi.solver as solvmod
import steps.utilities.geom_decompose as gd
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng

import numpy as np
import time
import datetime

from tol_funcs import *

########################################################################

def setup_module():
    global rng, NITER, DT, INT, NINJECT, DCST, tolerance, MESHFILE

    rng = srng.create('r123', 1024) 
    rng.initialize(1000) # The max unsigned long

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
    dif_X.setDcst(DCST)
    
    return mdl

########################################################################

def gen_geom():
    mesh = smeshio.loadMesh('validation_rd_mpi/meshes/'+MESHFILE)[0]
    
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

def test_unbdiff2D():
    "Surface Diffusion - Unbounded, point source (Parallel TetOpSplit)"

    m = gen_model()
    g, patch_tris, patch_tris_n, ctri_idx, trirads, triareas = gen_geom()

    tet_hosts = gd.binTetsByAxis(g, steps.mpi.nhosts)
    tri_hosts = gd.partitionTris(g, tet_hosts, patch_tris)
    sim = solvmod.TetOpSplit(m, g, rng, False, tet_hosts, tri_hosts)
    
    tpnts = np.arange(0.0, INT, DT)
    ntpnts = tpnts.shape[0]

    #Create the big old data structure: iterations x time points x concentrations
    res_count = np.zeros((NITER, ntpnts, patch_tris_n))
    res_conc = np.zeros((NITER, ntpnts, patch_tris_n))

    for j in range(NITER):
        sim.reset()
        sim.setTriCount(ctri_idx, 'X', NINJECT)
        for i in range(ntpnts):
            sim.run(tpnts[i])
            for k in range(patch_tris_n):
                res_count[j, i, k] = sim.getTriCount(patch_tris[k], 'X')
                res_conc[j, i, k] = sim.getTriCount(patch_tris[k], 'X')/sim.getTriArea(patch_tris[k])

    itermeans_count = np.mean(res_count, axis = 0)
    itermeans_conc = np.mean(res_conc, axis = 0)


    tpnt_compare = [12, 16, 20]
    passed = True
    max_err = 0.0

    for t in tpnt_compare:
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
                assert tolerable(det_conc, steps_conc, tolerance)
                
########################################################################
# END
