########################################################################

# Test endocytosis

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

########################################################################

class VesicleEndocytosis(unittest.TestCase):
    def test_endocytosis(self):
        # Simulation parameters
        scale = 1e-6

        ########################################################################

        # First order irreversible parameters
        NITER = 1000
        KCST = 10
        spec_N = 10

        ########################################################################

        # Raft and vesicle-related parameters
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
            ssys = SurfaceSystem()
            spec = Species.Create()
            vesicle = Vesicle.Create(vesicle_diam, DCST)

            with ssys:
                endo = Endocytosis.Create(vesicle, KCST, spec, True)

        ########################################################################

        mesh = TetMesh.LoadAbaqus(os.path.join(FILEDIR, 'meshes/sphere_0.5D_577tets.inp'), scale)

        with mesh:
            cyto = Compartment.Create(mesh.tets)
            memb = Patch.Create(mesh.surface, cyto, None, ssys)

            with memb:
                for tri in memb.tris:
                    EndocyticZone([tri])

        ########################################################################

        rng = RNG('mt19937', 512, 100)

        sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)

        rs = ResultSelector(sim)

        ves_count = rs.cyto.vesicle.Count
        spec_count = rs.memb.spec.Count

        sim.toSave(ves_count, spec_count, dt=DT)

        filePrefix = os.path.join(FILEDIR, 'data/endocytosis_test')
        if MPI.rank == 0 and os.path.isfile(f'{filePrefix}.h5'):
            os.remove(f'{filePrefix}.h5')

        with HDF5Handler(filePrefix) as hdf:
            sim.toDB(hdf, 'endocytosis')

            for i in range(NITER):
                if MPI.rank == 0:
                    print(i + 1, 'of', NITER)

                sim.newRun()

                sim.memb.spec.Count = spec_N

                sim.run(INT)

        if MPI.rank == 0:
            with HDF5Handler(filePrefix) as hdf:
                ves_count, spec_count = hdf['endocytosis'].results

                tpnts = spec_count.time[0]
                mean_res = np.mean(spec_count.data, axis=0).flatten()
                std_res = np.std(spec_count.data, axis=0).flatten()

                analy = spec_N * np.exp(-KCST * tpnts)
                std = np.sqrt(spec_N * np.exp(-KCST * tpnts) * (1 - np.exp(-KCST * tpnts)))

                plt.errorbar(
                    tpnts,
                    analy,
                    std,
                    color='black',
                    label='analytical',
                    linewidth=LINEWIDTH,
                )
                p = plt.errorbar(
                    tpnts + DT / 3,
                    mean_res,
                    std_res,
                    color='cyan',
                    ls='--',
                    label='STEPS',
                    linewidth=LINEWIDTH,
                )
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Dep. species number')
                fig = plt.gcf()
                fig.set_size_inches(3.4, 3.4)
                fig.savefig(os.path.join(FILEDIR, 'plots/endocytosis_spec.pdf'), dpi=300, bbox_inches='tight')
                plt.close()


                tpnts = ves_count.time[0]
                mean_res_v = np.mean(ves_count.data, axis=0).flatten()
                analy_v = spec_N * (1 - np.exp(-KCST * tpnts))

                plt.plot(
                    tpnts,
                    analy_v,
                    color='black',
                    label='analytical',
                    linewidth=LINEWIDTH,
                )
                plt.plot(
                    tpnts,
                    mean_res_v,
                    'c--',
                    label='STEPS',
                    linewidth=LINEWIDTH,
                )
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Vesicle number')
                fig = plt.gcf()
                fig.set_size_inches(3.4, 3.4)
                fig.savefig(os.path.join(FILEDIR, 'plots/endocytosis_ves.pdf'), dpi=300, bbox_inches='tight')
                plt.close()

                self.assertTrue(np.allclose(analy.flatten(), mean_res.flatten(), rtol=0.05))
                self.assertTrue(np.allclose(std.flatten(), std_res.flatten(), rtol=0.05))
                self.assertTrue(np.allclose(analy_v.flatten(), mean_res_v.flatten(), rtol=0.05))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(VesicleEndocytosis))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
