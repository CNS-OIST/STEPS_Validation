########################################################################

# Test endocytosis rate of Rafts

########################################################################

import steps.interface

from steps.model import *
from steps.geom import *
from steps.rng import *
from steps.sim import *
from steps.saving import *

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import unittest

FILEDIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

class VesicleRaftEndocytosis(unittest.TestCase):
    def test_raftendocytosis(self):
        ########################################################################

        # Simulation parameters
        scale = 1e-6

        ########################################################################

        # First order irreversible parameters
        NITER = 1000
        KCST = 20

        # Raft and vesicle-related parameters
        raft_N = 20
        raft_diam = 10e-9
        vesicle_diam = 5e-9
        DCST = 1e-12

        ########################################################################

        INT = 0.31
        DT = 0.01

        AVOGADRO = 6.022e23
        LINEWIDTH = 2

        ########################################################################

        model = Model()

        with model:
            rssys = RaftSurfaceSystem.Create()
            vesicle1 = Vesicle.Create(vesicle_diam, DCST)
            raft1 = Raft.Create(raft_diam, 0.0, rssys)
            with rssys:
                raftendo1 = RaftEndocytosis.Create(vesicle1, KCST)

            spec = Species.Create()  # sim needs one species

        ########################################################################

        mesh = TetMesh.LoadAbaqus(os.path.join(FILEDIR, 'meshes/sphere_0.5D_577tets.inp'), scale)

        with mesh:
            cyto = Compartment.Create(mesh.tets)
            memb = Patch.Create(mesh.surface, cyto)

        ########################################################################

        rng = RNG('mt19937', 512, 100)

        sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)

        rs = ResultSelector(sim)

        raft1_count = rs.memb.raft1.Count

        sim.toSave(raft1_count, dt=DT)

        filePrefix = os.path.join(FILEDIR, 'data/raftendocytosis_test')
        if MPI.rank == 0 and os.path.isfile(f'{filePrefix}.h5'):
            os.remove(f'{filePrefix}.h5')

        with HDF5Handler(filePrefix) as hdf:
            sim.toDB(hdf, f'raftendocytosis')

            for i in range(0, NITER):
                if MPI.rank == 0:
                    print(i + 1, 'of', NITER)

                sim.newRun()
                sim.memb.raft1.Count = raft_N

                sim.run(INT)

        if MPI.rank == 0:
            with HDF5Handler(filePrefix) as hdf:
                raft1_count, = hdf['raftendocytosis'].results
                tpnts = raft1_count.time[0]
                mean_res = np.mean(raft1_count.data, axis=0).flatten()
                std_res = np.std(raft1_count.data, axis=0).flatten()

                analy = raft_N * np.exp(-KCST * tpnts)
                std = np.sqrt(raft_N * np.exp(-KCST * tpnts)
                              * (1 - np.exp(-KCST * tpnts)))

                plt.errorbar(
                    tpnts,
                    analy,
                    std,
                    color='black',
                    label='analytical',
                    linewidth=LINEWIDTH,
                )
                plt.errorbar(
                    tpnts + DT / 3.0,
                    mean_res,
                    std_res,
                    ls='--',
                    label='STEPS',
                    linewidth=LINEWIDTH,
                )
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Raft number')
                fig = plt.gcf()
                fig.set_size_inches(3.4, 3.4)
                fig.savefig(os.path.join(FILEDIR, 'plots/raftendocytosis.pdf'), dpi=300, bbox_inches='tight')
                plt.close()

                self.assertTrue(np.allclose(analy, mean_res, rtol=0.05, atol=0.1))
                self.assertTrue(np.allclose(std, std_res, rtol=0.05, atol=0.1))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(VesicleRaftEndocytosis))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    # If the script is run manually, use the same endtime as for the paper
    INT = 100000.1
    unittest.TextTestRunner(verbosity=20).run(suite())
