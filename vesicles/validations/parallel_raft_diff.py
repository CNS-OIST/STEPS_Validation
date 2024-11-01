########################################################################

# surface diffusion of rafts on mesh surface from central source.

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
import os
import unittest

FILEDIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

class VesicleRaftDiff(unittest.TestCase):
    def test_raftdiff(self):
        ########################################################################

        # Number of iterations; plotting dt; sim endtime:
        NITER = 1000

        DT = 0.01
        INT = 0.16

        # Number of molecules injected in center; diff constant; number of tets sampled:
        NINJECT = 100

        DCST = 10e-12

        raft_diam = 1.0e-8

        ########################################################################

        model = Model()

        with model:
            ssys = SurfaceSystem.Create()

            raft = Raft.Create(raft_diam, DCST)
            X = Species.Create()

            with ssys:
                Diffusion(X, 0)

        ########################################################################

        mesh = TetMesh.Load(os.path.join(FILEDIR, f'meshes/coin_10r_1h_13861'))

        with mesh:
            cyto = Compartment.Create(mesh.tets)

            patch_tris = TriList(
                [tri for tri in mesh.surface if all(v.z > 0 for v in tri.verts)])

            memb = Patch.Create(patch_tris, cyto, None, ssys)

            ctri = (patch_tris & mesh.tets[0, 0, 0.5e-6].faces)[-1]

            trirads = [np.linalg.norm(tri.center - ctri.center) for tri in memb.tris]
            triareas = [tri.Area for tri in memb.tris]

        ########################################################################

        rng = RNG('r123', 1024, 100)
        sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE)

        rs = ResultSelector(sim)

        rafts = rs.TRIS(memb.tris).raft.Count

        sim.toSave(rafts, dt=DT)

        filePrefix = os.path.join(FILEDIR, 'data/raft_diff_test')
        if MPI.rank == 0 and os.path.isfile(f'{filePrefix}.h5'):
            os.remove(f'{filePrefix}.h5')

        with HDF5Handler(filePrefix) as hdf:
            sim.toDB(hdf, f'raft_diff')
            for j in range(NITER):
                if MPI.rank == 0:
                    print(j, 'of', NITER)
                sim.newRun()
                sim.TRI(ctri).raft.Count = NINJECT

                sim.run(INT)

        if MPI.rank == 0:
            tpnt_compare = range(3, 13, 2)

            with HDF5Handler(filePrefix) as hdf:
                rafts, = hdf['raft_diff'].results
                first = True
                concs = []
                for t in tpnt_compare:
                    bin_n = 20

                    bins = np.histogram_bin_edges(trirads, bin_n)
                    tri_bins = np.digitize(trirads, bins)

                    # Compute average position for each bin
                    bins_pos = np.bincount(tri_bins,
                                           weights=trirads) / np.bincount(tri_bins)
                    bin_areas = np.bincount(tri_bins, weights=triareas)

                    bin_counts = np.bincount(tri_bins,
                                             weights=np.mean(rafts.data[:, t, :], axis=0))
                    bin_concs = bin_counts / (bin_areas * 1e12)

                    det_conc = 1.0e-12 * (NINJECT /
                                          (4 * np.pi * DCST * rafts.time[0, t])) * (np.exp(
                                              (-1.0 * (bins_pos**2)) /
                                              (4 * DCST * rafts.time[0, t])))

                    concs.append((det_conc[1:], bin_concs[1:]))

                    if first:
                        plt.plot(bins_pos * 1e6,
                                 det_conc,
                                 'k-',
                                 label='analytical',
                                 linewidth=3)
                        plt.plot(bins_pos * 1e6,
                                 bin_concs,
                                 'r--',
                                 label='STEPS',
                                 linewidth=3)
                        first = False
                    else:
                        plt.plot(bins_pos * 1e6, det_conc, 'k-', linewidth=3)
                        plt.plot(bins_pos * 1e6, bin_concs, 'r--', linewidth=3)

                plt.xlabel('Distance from origin ($\mu$m)')
                plt.ylabel('Raft density ($\mu$m$^{-2}$)')
                plt.legend()
                plt.xlim(0, 5)
                fig = plt.gcf()
                fig.set_size_inches(3.4, 3.4)
                fig.savefig(os.path.join(FILEDIR, "plots/raft_diff.pdf"), dpi=300, bbox_inches='tight')
                plt.close()

                for det_conc, bin_concs in concs:
                    self.assertTrue(np.allclose(det_conc, bin_concs, rtol=0.05, atol=0.1))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(VesicleRaftDiff))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
