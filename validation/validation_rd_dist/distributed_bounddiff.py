########################################################################

# This is the parallel version of boundiff.py in validation_rd.
# Use for parallel TetOpSplit validation.

########################################################################

import steps.interface

import math
import numpy
import unittest

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from . import tol_funcs
from ..config import Configuration

configuration = Configuration(__file__)

########################################################################

class TestBoundDiff(unittest.TestCase):

    def test_bounddiff(self):
        "Diffusion - Bounded (Parallel TetOpSplit)"

        NITER = 10
        DT = 0.01
        INT = 0.11

        # The number of initial molecules:
        NINJECT = 10000	

        DCST = 0.2e-9

        # In tests, with good code, <1% fail with a tolerance of 5%
        tolerance = 5.0/100

        # The number of tets to sample at random:
        SAMPLE = 1060

        MESHFILE = 'cyl_diam2__len10_1060tets.msh'

    ########################################################################

        # Model
        mdl = Model()
        with mdl:
            X = Species.Create()
            cytosolv = VolumeSystem.Create()

            with cytosolv:
                dif_X = Diffusion.Create(X, DCST)

    ########################################################################

        # Mesh
        mesh = DistMesh(configuration.mesh_path(MESHFILE))

        assert(SAMPLE == len(mesh.tets))
        
        a = mesh.bbox.max.z-mesh.bbox.min.z
        area = sum(tet.Vol for tet in mesh.tets) / a
        
        with mesh:
            comp = Compartment.Create(cytosolv, tetLst=mesh.tets)

            # create the array of tet indices
            tets = mesh.tets[:SAMPLE]
            # Now find the distance of the centre of the tets to the Z lower face
            tetrads = numpy.array([(tet.center.z - mesh.bbox.min.z) * 1e6 for tet in tets])
            tetvols = numpy.array([tet.Vol for tet in tets])

            boundzmin = mesh.bbox.min.z + 0.01e-6
            minztets = TetList(tri.tetNeighbs[0] for tri in mesh.surface if all(v.z < boundzmin for v in tri.verts))
        
    ########################################################################

        rng = RNG('mt19937', 512, 1000)

        sim = Simulation('DistTetOpSplit', mdl, mesh, rng)

        rs = ResultSelector(sim)
        res = rs.TETS(tets).X.Count

        sim.toSave(res, dt=DT)

        conc = NITER * 6.022e23 * 1.0e-3 / minztets.Vol

        for j in range(NITER):
            sim.newRun()

            sim.TETS(minztets).X.Count = int(NINJECT / len(minztets))

            sim.run(INT)

        ########################################################################

        if MPI.rank == 0:
            counts = numpy.mean(res.data, axis=0)

            D = DCST
            pi = math.pi
            nmax = 1000
            N = NINJECT
            N = int((1.0*NINJECT)/len(minztets))*len(minztets)
            def getprob(x,t):
                p=0.0
                for n in range(nmax):
                    A = math.sqrt((1.0 if n == 0 else 2.0) / a)
                    p+= math.exp(-D*math.pow((n*pi/a), 2)*t)*A*math.cos(n*pi*x/a)*A*a
                return p*N/a

            NBINS = 5
            bins = numpy.histogram_bin_edges(tetrads, bins=NBINS)
            dig = numpy.digitize(tetrads, bins)

            bin_vols = numpy.bincount(dig, weights=tetvols)
            with numpy.errstate(invalid='ignore'):
                bin_rads = numpy.bincount(dig, weights=tetrads) / numpy.bincount(dig)

            tpnt_compare = [6, 8, 10]
            passed = True
            max_err = 0.0

            for tind in tpnt_compare:

                t = res.time[0, tind]

                bin_counts = numpy.bincount(dig, weights=counts[tind,:])
                with numpy.errstate(invalid='ignore'):
                    bin_conc = bin_counts / bin_vols * (1.0e-3/6.022e23)*1.0e6
                
                for r, conc in zip(bin_rads, bin_conc):
                    if 2 < r < 8:
                        rad = r * 1e-6
                        det_conc = (getprob(rad, t)/area)/6.022e20
                        assert(tol_funcs.tolerable(det_conc, conc, tolerance))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.makeSuite(TestBoundDiff, "test"))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
