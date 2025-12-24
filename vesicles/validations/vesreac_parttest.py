########################################################################

# Test 4 different types of vesicle surface reaction: 2 first-order
# reactions and 3 second-order reactions interacting with cytosolic species

########################################################################

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import unittest

FILEDIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

record_mode = False

class VesicleVesReac(unittest.TestCase):
    def test_vesreac_n4(self):
        ########################################################################

        # Simulation parameters
        scale = 1e-6

        ########################################################################

        # Vesicle-related parameters
        ves_N = 10
        ves_diam = 40e-9

        ########################################################################

        # First order irreversible parameters
        NITER_foi = 1000
        KCST_foi = 10
        spec_A_foi_number_perves = 10
        spec_A_foi_N = spec_A_foi_number_perves * ves_N

        # First order reversible parameters
        NITER_for = 10
        KCST_f_for = 100.0
        KCST_b_for = 20.0
        spec_A_for_number_perves = 50
        spec_A_for_N = spec_A_for_number_perves * ves_N

        # Second order irreversible AA parameters
        NITER_soAA = 1000
        KCST_soAA = 2e6
        spec_A_soAA_number_perves = 100
        spec_B_soAA_number_incomp = ves_N * spec_A_soAA_number_perves

        # Second order irreversible AB parameters
        NITER_soAB = 1000
        KCST_soAB = 0.5e6
        n_soAB = 2
        spec_A_soAB_number_perves = 50
        spec_B_soAB_number_incomp = ves_N * spec_A_soAB_number_perves / n_soAB

        ########################################################################

        NITER_max = 1
        INT = 10.1
        DT = 0.1

        AVOGADRO = 6.022e23
        LINEWIDTH = 3

        ########################################################################

        model = Model()
        r = ReactionManager()

        with model:
            vsys = VolumeSystem.Create()
            vssysa = VesicleSurfaceSystem.Create()
            vssysb = VesicleSurfaceSystem.Create()
            vssysc = VesicleSurfaceSystem.Create()
            vssysd = VesicleSurfaceSystem.Create()

            ves = Vesicle.Create(ves_diam, 1e-12, vssysa)
            ves.addSystem(vssysb)
            ves.addSystem(vssysc)
            ves.addSystem(vssysd)

            A_foi, A_for, B_for, A_soAA, B_soAA, C_soAA, A_soAB, B_soAB, C_soAB = Species.Create(
            )

            with vssysa:
                # First order irreversible
                A_foi.v > r[1] > A_foi.v
                r[1].K = KCST_foi

            with vssysb:
                # First order reversible
                A_for.v < r[1] > B_for.v
                r[1].K = KCST_f_for, KCST_b_for

            with vssysc:
                # Second order irreversible AA
                B_soAA.o + A_soAA.v < r[1] > C_soAA.v
                r[1].K = KCST_soAA, KCST_foi

            with vssysd:
                # Second order irreversible AB
                B_soAB.o + A_soAB.v < r[1] > C_soAB.v
                r[1].K = KCST_soAB, KCST_foi

            with vsys:
                Diffusion(B_soAA, 10e-12)
                Diffusion(B_soAB, 10e-12)

        ########################################################################

        mesh = TetMesh.LoadAbaqus(os.path.join(FILEDIR, 'meshes/sphere_0.5D_2088tets.inp'), scale)

        with mesh:
            acomptets = TetList(tet for tet in mesh.tets if tet.center.x < 0 and tet.center.y < 0)
            bcomptets = TetList(tet for tet in mesh.tets if tet.center.x > 0 and tet.center.y < 0)
            ccomptets = TetList(tet for tet in mesh.tets if tet.center.x < 0 and tet.center.y > 0)
            dcomptets = mesh.tets - acomptets - bcomptets - ccomptets
            
            compa = Compartment.Create(acomptets, vsys)
            compb = Compartment.Create(bcomptets, vsys)
            compc = Compartment.Create(ccomptets, vsys)
            compd = Compartment.Create(dcomptets, vsys)

        ########################################################################

        rng = RNG('mt19937', 512, 100)
        use_partition=True
        
        if record_mode:
            sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)
            #sim.autoWeightLog(period=0.0001, prefix="prop_weighted/", method='avg-propensities')
            sim.autoWeightLog(period=0.1, prefix="prop_weighted/", method='extents')
        else:
            if use_partition:
                partition = TetWeightPartition(mesh, prefix="prop_weighted/", n_hosts=8, start_host=1)
                if MPI.rank ==0: partition.printStats()
                sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, tet_hosts=partition._tet_hosts)
            else: sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)


        CONCA_soAA = (ves_N * spec_A_soAA_number_perves) / (AVOGADRO * compc.Vol * 1e3)
        CONCB_soAA = spec_B_soAA_number_incomp / (AVOGADRO * compc.Vol * 1e3)

        CONCA_soAB = (ves_N * spec_A_soAB_number_perves) / (AVOGADRO * compd.Vol * 1e3)
        CONCB_soAB = CONCA_soAB / n_soAB

        rs = ResultSelector(sim)


        filePrefix = os.path.join(FILEDIR, 'data/vesreac_test')
        if MPI.rank == 0 and os.path.isfile(f'{filePrefix}.h5'):
            os.remove(f'{filePrefix}.h5')

        with HDF5Handler(filePrefix) as hdf:
            sim.toDB(hdf, f'vesreac')
            for i in range(NITER_max):
                if MPI.rank == 0:
                    print(i, 'of', NITER_max)
                
                btime=time.time()
                
                sim.newRun()
                
                
                sim.compa.ves.Count = ves_N/4
                sim.compb.ves.Count = ves_N/4
                sim.compc.ves.Count = ves_N/4
                sim.compd.ves.Count = ves_N/4

                if i < NITER_foi:
                    sim.compa.VESICLES()('surf').A_foi.Count = spec_A_foi_number_perves
                    sim.compb.VESICLES()('surf').A_foi.Count = spec_A_foi_number_perves
                    sim.compc.VESICLES()('surf').A_foi.Count = spec_A_foi_number_perves
                    sim.compd.VESICLES()('surf').A_foi.Count = spec_A_foi_number_perves
                if i < NITER_for:
                    sim.compa.VESICLES()('surf').A_for.Count = spec_A_for_number_perves
                    sim.compb.VESICLES()('surf').A_for.Count = spec_A_for_number_perves
                    sim.compc.VESICLES()('surf').A_for.Count = spec_A_for_number_perves
                    sim.compd.VESICLES()('surf').A_for.Count = spec_A_for_number_perves
                if i < NITER_soAA:
                    sim.compa.B_soAA.Count = spec_B_soAA_number_incomp
                    sim.compa.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves
                    sim.compb.B_soAA.Count = spec_B_soAA_number_incomp
                    sim.compb.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves
                    sim.compc.B_soAA.Count = spec_B_soAA_number_incomp
                    sim.compc.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves
                    sim.compd.B_soAA.Count = spec_B_soAA_number_incomp
                    sim.compd.VESICLES()('surf').A_soAA.Count = spec_A_soAA_number_perves
                    
                if i < NITER_soAB:
                    sim.compa.B_soAB.Count = spec_B_soAB_number_incomp
                    sim.compa.VESICLES()('surf').A_soAB.Count = spec_A_soAB_number_perves
                    sim.compb.B_soAB.Count = spec_B_soAB_number_incomp
                    sim.compb.VESICLES()('surf').A_soAB.Count = spec_A_soAB_number_perves
                    sim.compc.B_soAB.Count = spec_B_soAB_number_incomp
                    sim.compc.VESICLES()('surf').A_soAB.Count = spec_A_soAB_number_perves
                    sim.compd.B_soAB.Count = spec_B_soAB_number_incomp
                    sim.compd.VESICLES()('surf').A_soAB.Count = spec_A_soAB_number_perves


                sim.run(INT)
                
                print ("took ", time.time()-btime, "s")


########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(VesicleVesReac))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    # If the script is run manually, use the same endtime as for the paper
    INT = 100000.1
    unittest.TextTestRunner(verbosity=20).run(suite())
