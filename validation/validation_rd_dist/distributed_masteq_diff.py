########################################################################

# This is the parallel version of masteq_diff.py in validation_rd.
# Use for parallel TetOpSplit validation.

########################################################################

import steps.interface

import math
import numpy
import unittest

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

from . import tol_funcs
from ..config import Configuration

configuration = Configuration(__file__)

########################################################################

class TestMasteqDiff(unittest.TestCase):

    def test_masteq_diff(self):
        "Reaction-diffusion - Production and second order degradation (Parallel DistTetOpSplit)"

        ### NOW   A+B-> B,  0->A (see Erban and Chapman, 2009)

        ########################################################################
        SCALE = 1.0

        KCST_f = 100e6*SCALE			# The reaction constant, degradation
        KCST_b = (20.0e-10*SCALE) 		# The reaction constant, production

        DCST_A = 20e-12
        DCST_B = 20e-12

        B0 = 1   # The number of B moleucles

        DT = 0.1			# Sampling time-step
        INT = 50000.1 		# Sim endtime

        filename = 'cube_1_1_1_73tets.msh'

        # A tolerance of 7.5% will fail <1% of the time
        tolerance = 7.5/100

        ########################################################################

        mdl = Model()
        r = ReactionManager()
        with mdl:
            SA, SB = Species.Create()
            volsys = VolumeSystem.Create()

            with volsys:
                diffA =         Diffusion.Create(SA, DCST_A)
                diffB =         Diffusion.Create(SB, DCST_B)

                # Production
                SA + SB >r['R1']> SB
                r['R1'].K = KCST_f

                None >r['R2']> SA
                r['R2'].K = KCST_b

        mesh = DistMesh(configuration.mesh_path(filename))
        with mesh:
            comp1 = Compartment.Create(volsys, tetLst = mesh.tets)

        rng = RNG('mt19937', 512, 1000)
        sim = Simulation('DistTetOpSplit', mdl, mesh, rng)

        rs = ResultSelector(sim)

        resA = rs.comp1.SA.Count

        sim.toSave(resA, dt=DT)

        sim.newRun()

        sim.comp1.SA.Count = 0
        sim.comp1.SB.Count = B0

        sim.run(INT)

        def fact(x): return (1 if x==0 else x * fact(x-1))

        # Do cumulative count, but not comparing them all. 
        # Don't get over 50 (I hope)
        steps_n_res = numpy.zeros(50)
        for r in resA.data[0,:,0]: steps_n_res[int(r)]+=1
        for s in range(50): steps_n_res[s] = steps_n_res[s]/len(resA.time[0])

        k1 = KCST_f / 6.022e23
        k2 = KCST_b * 6.022e23
        v = comp1.Vol * 1.0e3 # litres

        if MPI.rank == 0:
            for m in range(5, 11):
                analy = (1.0/fact(m))*math.pow((k2*v*v)/(B0*k1), m)*math.exp(-((k2*v*v)/(k1*B0)))
                assert(tol_funcs.tolerable(steps_n_res[m], analy, tolerance))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(TestMasteqDiff))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
