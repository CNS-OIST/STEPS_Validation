########################################################################

# 3D diffusion in an infinite volume from a point source. 

# AIMS: to verify STEPS spatial-deterministic solver 'TetODE' supports 
# local initial conditions and calculates volume diffusion rates correctly.

# STEPS equivalent model: Deterministic 3D diffusion in a sphere; source 
# from central tetrahedron.

# For a more detailed description of the analytical system and 
# equivalent STEPS model see:
# http://www.biomedcentral.com/content/supplementary/1752-0509-6-36-s4.pdf
# "3D diffusion in infinite volume"

# Verification also takes place of model and mesh construction 
# components, particularly mesh loading and manipulation capabilities 
# with functions such as steps.utilities.meshio.loadMesh and 
# steps.geom.Tetmesh.findTetByPoint, steps.geom.Tetmesh.getTetVol etc. 
# Localised recording by steps.solver.TetODE.getTetCount is also verified. 

# Even though this is a deterministic model, a tolerance of 3.5% is 
# permitted. This is because a point source is not replicated in STEPS 
# and a small error is introduced by the initial variance from the  
# tetrahedral source.
  
########################################################################

import steps.model as smodel
import steps.solver as solvmod
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng

import datetime
import time
import math
import numpy

from tol_funcs import *

########################################################################

def setup_module():
    global rng

    rng = srng.create('r123', 1024) 
    rng.initialize(1000) # The max unsigned long

    # Number of iterations; plotting dt; sim endtime:
    global NITER, DT, INT

    NITER = 1

    DT = 0.01
    INT = 0.21

    # Number of molecules injected in centre; diff constant; number of tets sampled:
    global NINJECT, DCST, tolerance

    NINJECT = 100000 	

    DCST = 0.02e-9

    # Small expected error from non point source
    tolerance = 3.5/100

    ########################################################################
    global SAMPLE, MESHFILE, tetidxs, tetrads, tetvols

    SAMPLE = 10000	 

    MESHFILE = 'sphere_rad10_77Ktets'

    # create the array of tet indices to be found at random
    tetidxs = numpy.zeros(SAMPLE, dtype = 'int')
    for i in range(SAMPLE): tetidxs[i] = i

    # further create the array of tet barycentre distance to centre
    tetrads = numpy.zeros(SAMPLE)
    tetvols = numpy.zeros(SAMPLE)

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
    mesh = smeshio.loadMesh('validation_rd/meshes/'+MESHFILE)[0]
    
    ctetidx = mesh.findTetByPoint([0.0, 0.0, 0.0])
    
    ntets = mesh.countTets()
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    comp.addVolsys('cytosolv')
    
    # Now fill the array holding the tet indices to sample at random
    assert(SAMPLE <= ntets)
    assert(SAMPLE >= 5)
    
    # First I'm going to add the centre tet and it's 4 neighbours
    ctetidx = mesh.findTetByPoint([0.0, 0.0, 0.0])
    tetidxs[0] = ctetidx
    neighbs = mesh.getTetTetNeighb(ctetidx)
    tetidxs[1] = neighbs[0]
    tetidxs[2] = neighbs[1]
    tetidxs[3] = neighbs[2]
    tetidxs[4] = neighbs[3]
    numfilled = 5    
    
    while (numfilled < SAMPLE):
        #idx = int((rng.getUnfII()*ntets)%ntets)
        max = mesh.getBoundMax()
        min = mesh.getBoundMin()
        
        maxX3 = 0.0
        maxY3 = 0.0
        maxZ3 = 0.0
        
        if (max[0] > -min[0]) : maxX3 = abs(math.pow(max[0], 1.0/3))
        else : maxX3 = abs(math.pow(abs(min[0]), 1.0/3))
        if (max[1] > -min[1]) : maxY3 = abs(math.pow(max[1], 1.0/3))
        else : maxY3 = abs(math.pow(abs(min[1]), 1.0/3))
        if (max[2] > -min[2]) : maxZ3 = abs(math.pow(max[2], 1.0/3))
        else : maxZ3 = abs(math.pow(abs(min[2]), 1.0/3))
        
        rnx = rng.getUnfII()
        rny = rng.getUnfII()
        rnz = rng.getUnfII()
        
        signx = rng.getUnfII()
        signy = rng.getUnfII()
        signz = rng.getUnfII()
        
        if (signx >= 0.5) : xpnt = math.pow((maxX3*rnx), 3)
        else : xpnt = -(math.pow((maxX3*rnx), 3))
        
        if (signy >= 0.5) : ypnt = math.pow((maxY3*rny), 3)
        else : ypnt = -(math.pow((maxY3*rny), 3))
        
        if (signz >= 0.5) : zpnt = math.pow((maxZ3*rnz), 3)
        else : zpnt = -(math.pow((maxZ3*rnz), 3))
        
        idx = mesh.findTetByPoint([xpnt, ypnt, zpnt])
        
        if (idx == -1): continue
        if (idx not in tetidxs): 
            tetidxs[numfilled] = idx
            numfilled += 1
    
    tetidxs.sort()
        
    # Now find the distance of the centre of the tets to the centre of the centre tet (at 0,0,0)
    cbaryc = mesh.getTetBarycenter(ctetidx)
    for i in range(SAMPLE):
        baryc = mesh.getTetBarycenter(int(tetidxs[i]))
        r2 = math.pow((baryc[0]-cbaryc[0]),2) + math.pow((baryc[1]-cbaryc[1]),2) + math.pow((baryc[2]-cbaryc[2]),2)
        r = math.sqrt(r2)
        # Conver to microns
        tetrads[i] = r*1.0e6
        tetvols[i] = mesh.getTetVol(int(tetidxs[i]))
    
    return mesh

########################################################################

def test_unbdiff_ode():
    "Diffusion - Unbounded (TetODE)"

    m = gen_model()
    g = gen_geom()

    # Fetch the index of the centre tet
    ctetidx = g.findTetByPoint([0.0, 0.0, 0.0])
    # And fetch the total number of tets to make the data structures
    ntets = g.countTets()

    sim = solvmod.TetODE(m, g)
    sim.setTolerances(1e-7, 1e-7)

    tpnts = numpy.arange(0.0, INT, DT)
    ntpnts = tpnts.shape[0]

    #Create the big old data structure: iterations x time points x concentrations
    res = numpy.zeros((NITER, ntpnts, SAMPLE))

    for j in range(NITER):
        sim.setTetCount(ctetidx, 'X', NINJECT)
        for i in range(ntpnts):
            sim.run(tpnts[i])
            for k in range(SAMPLE):
                res[j, i, k] = sim.getTetCount(int(tetidxs[k]), 'X')
                
    itermeans = numpy.mean(res, axis = 0)

    tpnt_compare = [10, 15, 20]
    passed = True
    max_err = 0.0

    for t in tpnt_compare:
        bin_n = 20
        
        r_max = tetrads.max()
        r_min = 0.0
        
        r_seg = (r_max-r_min)/bin_n
        bin_mins = numpy.zeros(bin_n+1)
        r_tets_binned = numpy.zeros(bin_n)
        bin_vols = numpy.zeros(bin_n)    
        
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
        
        bin_concs = numpy.zeros(bin_n)
        for c in range(bin_n): 
            for d in range(bin_counts[c].__len__()):
                bin_concs[c] += bin_counts[c][d]
            bin_concs[c]/=(bin_vols[c]*1.0e18)
        
        for i in range(bin_n):
            if (r_tets_binned[i] > 2.0 and r_tets_binned[i] < 6.0):
                rad = r_tets_binned[i]*1.0e-6
                det_conc = 1e-18*((NINJECT/(math.pow((4*math.pi*DCST*tpnts[t]),1.5)))*(math.exp((-1.0*(rad*rad))/(4*DCST*tpnts[t]))))
                steps_conc = bin_concs[i]
                assert tolerable(det_conc, steps_conc, tolerance)

########################################################################
# END
