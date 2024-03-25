########################################################################

# This is the parallel version of unbdiff2D_linesource_ring.py in validation_rd.
# Use for parallel TetOpSplit validation.

########################################################################

import steps.model as smodel
import steps.mpi
import steps.mpi.solver as solvmod
import steps.utilities.geom_decompose as gd
import steps.model as smodel
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng

import copy
import datetime
import numpy as np
import time
import unittest

from . import tol_funcs
from ..config import Configuration

configuration = Configuration(__file__)

########################################################################

rng = srng.create('r123', 1024) 
rng.initialize(1000) # The max 32bit unsigned long

# Number of iterations; plotting dt; sim endtime:
NITER = 100
DT = 0.02
INT = 3.02

# Number of molecules injected in centre; diff constant
NINJECT = 5000
DCST = 0.01e-9

# In tests <1% fail with tolerance of 5%
tolerance = 5.0/100

########################################################################

MESHFILE = 'ring2_or10_ir9_injx0width2_640tets.inp'

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
    mesh = smeshio.loadMesh(configuration.mesh_path(MESHFILE))[0]
        
    ntets = mesh.countTets()
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    
    alltris = mesh.getSurfTris()
    
    patch_tris = []
    for t in alltris:
        baryc = mesh.getTriBarycenter(t)
        rad = np.sqrt(np.power(baryc[0], 2)+np.power(baryc[1], 2))
        # By checking the cubit mesh the outer tris fall in the following bound
        if rad > 0.00000995 and rad < 0.00001001: patch_tris.append(t)    
                
    patch = stetmesh.TmPatch('patch', mesh, patch_tris, icomp = comp)
    patch.addSurfsys('ssys')
    
    partition_tris = copy.deepcopy(patch_tris)
    inject_tris=[]
    for p in patch_tris:
        # X should be maximum (10) for the inject region
        if   mesh.getTriBarycenter(p)[0] > 9.999e-6: 
            inject_tris.append(p)      
            
    for i in inject_tris: patch_tris.remove(i)
    patch_tris_n = len(patch_tris)

    # Now find the distances along the edge for all tris
    tridists = np.zeros(patch_tris_n)
    triareas = np.zeros(patch_tris_n)
    
    for i in range(patch_tris_n):
        baryc = mesh.getTriBarycenter(patch_tris[i])
        rad = np.sqrt(np.power(baryc[0], 2)+np.power(baryc[1], 2))
        theta = np.arctan2(baryc[1], baryc[0])
        
        tridists[i] = theta*rad*1e6
        triareas[i] = mesh.getTriArea(patch_tris[i])
    
    # Triangles will be separated into those to the 'high' side (positive y) with positive L
    # and those on the low side (negative y) with negative L 
    
    return mesh, patch_tris, partition_tris, patch_tris_n, inject_tris, tridists, triareas

########################################################################

class TestUnbDiff2DLineSourceRing(unittest.TestCase):

    def test_unbdiff2D_linesource_ring(self):
        "Surface Diffusion - Unbounded, line source (Parallel TetOpSplit)"

        m = gen_model()
        g, patch_tris, partition_tris, patch_tris_n, inject_tris, tridists, triareas = gen_geom()

        tet_hosts = gd.linearPartition(g, [1, 1, steps.mpi.nhosts])
        tri_hosts = gd.partitionTris(g, tet_hosts, partition_tris)

        sim = solvmod.TetOpSplit(m, g, rng, False, tet_hosts, tri_hosts)

        tpnts = np.arange(0.0, INT, DT)
        ntpnts = tpnts.shape[0]

        #Create the big old data structure: iterations x time points x concentrations
        res_count = np.zeros((NITER, ntpnts, patch_tris_n))
        res_conc = np.zeros((NITER, ntpnts, patch_tris_n))

        for j in range(NITER):
            sim.reset()
            for t in inject_tris:
                sim.setTriCount(t, 'X', float(NINJECT)/len(inject_tris))
            for i in range(ntpnts):
                sim.run(tpnts[i])
                for k in range(patch_tris_n):
                    res_count[j, i, k] = sim.getTriCount(patch_tris[k], 'X')
                    res_conc[j, i, k] = 1e-12*(sim.getTriCount(patch_tris[k], 'X')/sim.getTriArea(patch_tris[k]))

        itermeans_count = np.mean(res_count, axis = 0)
        itermeans_conc = np.mean(res_conc, axis = 0)

        ########################################################################

        tpnt_compare = [100, 150]

        passed = True
        max_err = 0.0

        for t in tpnt_compare:
            bin_n = 50
            
            r_min=0
            r_max=0
            
            for i in tridists: 		
                if (i > r_max): r_max = i 	
                if (i < r_min): r_min = i
            
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
                i_r = tridists[i]
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
                if (r_tris_binned[i] > -10.0 and r_tris_binned[i] < -2.0) \
                or (r_tris_binned[i] > 2.0 and r_tris_binned[i] < 10.0):
                    dist = r_tris_binned[i]*1e-6
                    det_conc = 1e-6*(NINJECT/(4*np.sqrt((np.pi*DCST*tpnts[t]))))*(np.exp((-1.0*(dist*dist))/(4*DCST*tpnts[t])))	
                    steps_conc = bin_concs[i]
                    assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(TestUnbDiff2DLineSourceRing))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
