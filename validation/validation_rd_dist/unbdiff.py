########################################################################

# This is the parallel version of unbdiff.py in validation_rd.
# Use for parallel TetOpSplit validation.

########################################################################

import steps.interface

import math
import numpy
from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from . import tol_funcs
from .. import configuration

########################################################################

def test_unbdiff():
    "Diffusion - Unbounded (Parallel TetOpSplit)"

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
    MESHFILE = 'sphere_rad10_33Ktets_adaptive.msh'
    
    # Model
    mdl = Model()
    with mdl:
        X = Species.Create()
        cytosolv = VolumeSystem.Create()

        with cytosolv:
            dif_X = Diffusion.Create(X, DCST)

    mesh = DistMesh(configuration.mesh_path(MESHFILE))

    # Fetch the index of the centre tet
    ctet = mesh.tets[0.0, 0.0, 0.0]

    with mesh:
        comp = Compartment.Create(cytosolv, tetLst=mesh.tets)

        # create the array of tet indices to be found at random
        tets = mesh.tets[:SAMPLE]
        tetrads = numpy.array([numpy.linalg.norm(tet.center)*1e6 for tet in tets])
        tetvols = numpy.array([tet.Vol for tet in tets])

    rng = RNG('mt19937', 512, 1000)

    sim = Simulation('DistTetOpSplit', mdl, mesh, rng)

    rs = ResultSelector(sim)
    res = rs.TETS(tets).X.Count

    sim.toSave(res, dt=DT)

    for j in range(NITER):
        sim.newRun()

        sim.TET(ctet).X.Count = NINJECT

        sim.run(INT)

    if MPI.rank == 0:
        counts = numpy.mean(res.data, axis=0)

        bin_n = 20
        bins = numpy.histogram_bin_edges(tetrads, bins=bin_n)
        dig = numpy.digitize(tetrads, bins)

        bin_vols = numpy.bincount(dig, weights=tetvols)
        with numpy.errstate(invalid='ignore'):
            bin_rads = numpy.bincount(dig, weights=tetrads) / numpy.bincount(dig)

        tpnt_compare = [10, 15, 20]
        passed = True
        max_err = 0.0

        for tind in tpnt_compare:
            t = res.time[0, tind]

            bin_counts = numpy.bincount(dig, weights=counts[tind,:])
            with numpy.errstate(invalid='ignore'):
                bin_conc = bin_counts / (bin_vols * 1e18)

            for r, conc in zip(bin_rads, bin_conc):
                if 2 < r < 6:
                    rad = r * 1e-6
                    det_conc = 1e-18*((NINJECT/(math.pow((4*math.pi*DCST*t),1.5)))*(math.exp((-1.0*(rad*rad))/(4*DCST*t))))
                    assert(tol_funcs.tolerable(det_conc, conc, tolerance))

########################################################################
# END
