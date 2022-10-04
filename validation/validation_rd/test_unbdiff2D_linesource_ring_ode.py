########################################################################

# 2D diffusion on an infinite strip from line "point source". 

# AIMS: to verify STEPS spatial-deterministic solver 'TetODE' supports 
# local initial conditions and calculates surface diffusion rates correctly.

# STEPS equivalent model: Deterministic 2D diffusion on outer surface of 
# ring; source in small region of x=10um triangles.

# Verification also takes place of model and mesh construction 
# components, particularly mesh loading and manipulation capabilities 
# with functions such as steps.geom.Tetmesh.getTriBarycenter and 
# steps.geom.Tetmesh.getTriArea etc. 
# Localised recording by steps.solver.TetODE.getTriCount is also verified. 

# Even though this is a deterministic model, a tolerance of 1.0% is 
# permitted. This is because a point source is not replicated in STEPS 
# and a small error is introduced by the initial variance from the  
# finite triangle source.
  
########################################################################

import steps.model as smodel
import steps.solver as solvmod
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng

import datetime
import time
import numpy as np
import unittest

from . import tol_funcs
from ..config import Configuration

configuration = Configuration(__file__)

########################################################################

# plotting dt; sim endtime:
DT = 0.02
INT = 3.02

# Number of molecules injected in centre; diff constant
NINJECT = 1000
DCST = 0.01e-9

# Error from non-zero injection width etc should be less than 1%
tolerance = 1.0/100

########################################################################

MESHFILE ='ring2_or10_ir9_injx0width2_60084tets.inp'

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
        if rad > 0.000009955 and rad < 0.00001001: patch_tris.append(t)    
    
    patch = stetmesh.TmPatch('patch', mesh, patch_tris, icomp = comp)
    patch.addSurfsys('ssys')
    
    area = 0
    for t in patch_tris: area+=mesh.getTriArea(t)

    inject_tris=[]
    for p in patch_tris:
        # X should be maximum (10) for the inject region
        if   mesh.getTriBarycenter(p)[0] > 9.999e-6: 
            inject_tris.append(p)      
        
    patch_tris_n = len(patch_tris)

    # Now find the distances along the edge for all tris
    tridists = np.zeros(patch_tris_n)
    triareas = np.zeros(patch_tris_n)
    
    for i in range(patch_tris_n):
        baryc = mesh.getTriBarycenter(patch_tris[i])
        rad = np.sqrt(np.power(baryc[0], 2)+np.power(baryc[1], 2))
        
        theta = np.arctan2(baryc[1], baryc[0])
        
        tridists[i] = (theta*rad*1e6)
        triareas[i] = mesh.getTriArea(patch_tris[i])
        
    # Triangles will be separated into those to the 'high' side (positive y) with positive L
    # and those on the low side (negative y) with negative L 

    return mesh, patch_tris, patch_tris_n, inject_tris, tridists, triareas

########################################################################

class TestUnbDiff2DLineSourceRingODE(unittest.TestCase):
    def test_unbdiff2D_linesource_ring_ode(self):
        "Surface Diffusion - Unbounded, line source (TetODE)"
        rng = srng.create('r123', 1024) 
        rng.initialize(1000) # The max unsigned long

        m = gen_model()
        g, patch_tris, patch_tris_n, inject_tris, tridists, triareas = gen_geom()


        sim = solvmod.TetODE(m, g, rng)
        sim.setTolerances(1e-7, 1e-7)

        tpnts = np.arange(0.0, INT, DT)
        ntpnts = tpnts.shape[0]

        res_count = np.zeros((ntpnts, patch_tris_n))

        for t in inject_tris:
            sim.setTriCount(t, 'X', float(NINJECT)/len(inject_tris))
        for i in range(ntpnts):
            sim.run(tpnts[i])
            for k in range(patch_tris_n):
                res_count[i, k] = sim.getTriCount(patch_tris[k], 'X')


        tpnt_compare = [75, 100, 150]

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
            for i in range((res_count[t].size)):
                i_r = tridists[i]
                for b in range(bin_n):
                    if(i_r>=bin_mins[b] and i_r<bin_mins[b+1]):
                        bin_counts[b].append(res_count[t][i])
                        bin_areas[b]+=sim.getTriArea(int(patch_tris[i]))
                        break
            
            bin_concs = np.zeros(bin_n)
            for c in range(bin_n): 
                for d in range(bin_counts[c].__len__()):
                    bin_concs[c] += bin_counts[c][d]
                bin_concs[c]/=(bin_areas[c]*1.0e12)
            
            for i in range(bin_n):
                if (r_tris_binned[i] > -10.0 and r_tris_binned[i] < 10.0):
                    dist = r_tris_binned[i]*1e-6
                    det_conc = 1e-6*(NINJECT/(4*np.sqrt((np.pi*DCST*tpnts[t]))))*(np.exp((-1.0*(dist*dist))/(4*DCST*tpnts[t])))	
                    steps_conc = bin_concs[i]
                    assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)
                    
########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.makeSuite(TestUnbDiff2DLineSourceRingODE, "test"))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
