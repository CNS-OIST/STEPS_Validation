########################################################################

# This is the parallel version of unbdiff.py in validation_rd.
# Use for parallel TetOpSplit validation.

########################################################################

from __future__ import print_function, absolute_import

import steps.model as smodel
import steps.mpi
import steps.mpi.solver as solvmod
import steps.utilities.meshio as smeshio
import steps.geom as stetmesh
import steps.rng as srng

import numpy
import math
import datetime
import time

from . import tol_funcs

########################################################################

def setup_module():
    global rng

    rng = srng.create('r123', 1024)
    rng.initialize(1000) # The max unsigned long

    global NITER, DT, INT, NINJECT, DCST, tolerance

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
    global SAMPLE, MESHFILE

    SAMPLE = 32552	 # all tets
    MESHFILE = 'sphere_rad10_33Ktets_adaptive'

    # create the array of tet indices to be found at random
    global tetidxs
    tetidxs = numpy.zeros(SAMPLE, dtype = 'int')
    for i in range(SAMPLE): tetidxs[i] = i

    # further create the array of tet barycentre distance to centre
    global tetrads, tetvols
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
    mesh = smeshio.loadMesh('validation_rd_mpi/meshes/'+MESHFILE)[0]

    ctetidx = mesh.findTetByPoint([0.0, 0.0, 0.0])

    ntets = mesh.countTets()
    comp = stetmesh.TmComp('cyto', mesh, range(ntets))
    comp.addVolsys('cytosolv')

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

def test_unbdiff():
    "Diffusion - Unbounded (Parallel TetOpSplit)"

    m = gen_model()
    g = gen_geom()

    # Fetch the index of the centre tet
    ctetidx = g.findTetByPoint([0.0, 0.0, 0.0])
    # And fetch the total number of tets to make the data structures
    ntets = g.countTets()

    sim = solvmod.TetOpSplit(m, g, rng, False)


    tpnts = numpy.arange(0.0, INT, DT)
    ntpnts = tpnts.shape[0]

    # Create the big old data structure: iterations x time points x concentrations
    res = numpy.zeros((NITER, ntpnts, SAMPLE))

    for j in range(NITER):
        sim.reset()
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
                assert tol_funcs.tolerable(det_conc, steps_conc, tolerance)

########################################################################
# END
