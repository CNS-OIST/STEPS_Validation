########################################################################

# Test paths

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


class VesiclePath(unittest.TestCase):
    def test_path(self):
        ########################################################################

        scale = 1e-6
        NTPNTS = 11000
        DT = 0.001

        ves_diam = 50e-9

        ########################################################################

        model = Model()

        with model:
            vsys = VolumeSystem.Create()
            ssys = SurfaceSystem.Create()
            vssys = VesicleSurfaceSystem.Create()

            ves1 = Vesicle.Create(ves_diam, 1e-12, vssys)
            raft1 = Raft.Create(ves_diam, 1e-12)
            spec1, spec2 = Species.Create()

            with vsys:
                spec1_vdiff = Diffusion.Create(spec1, 1e-12)

            with ssys:
                spec1_sdiff = Diffusion.Create(spec1, 0.1e-12)
                spec2_sdiff = Diffusion.Create(spec2, 0.1e-12)

            with vssys:
                exo = Exocytosis.Create(10000)

        ########################################################################

        mesh = TetMesh.LoadAbaqus(os.path.join(FILEDIR, 'meshes/sphere_0.5D_2088tets.inp'), scale)

        with mesh:
            comp1 = Compartment.Create(mesh.tets, vsys)

            tris_posz = TriList([tri for tri in mesh.surface if tri.center.z > 0.0])
            patch1 = Patch.Create(tris_posz, comp1, None, ssys)

        ########################################################################

        rng = RNG('mt19937', 512, 987)

        sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)

        speed1 = 0.5e-6
        speed2 = 0.3e-6

        actin1 = sim.addVesiclePath('actin1')
        actin1.addPoint([0, 0, -2.2e-7])
        actin1.addPoint([0, 0, 0])
        actin1.addPoint([0, 0, 2.2e-7])
        actin1.addBranch(1, {2: 1})
        actin1.addBranch(2, {3: 1})
        actin1.addVesicle(ves1,
                          speed=speed1,
                          dependencies=2 * spec1,
                          stoch_stepsize=1e-12)

        actin2 = sim.addVesiclePath('actin2')
        actin2.addPoint([0, 0, -2.2e-7])
        actin2.addPoint([0, 0, 0])
        actin2.addPoint([0, 0, 2.2e-7])
        actin2.addBranch(1, {2: 1})
        actin2.addBranch(2, {3: 1})
        actin2.addVesicle(ves1,
                          speed=speed2,
                          dependencies=2 * spec2,
                          stoch_stepsize=1e-10)

        rs = ResultSelector(sim)

        vesPos = rs.VESICLES(sim.comp1.ves1).Pos

        sim.toSave(vesPos, dt=DT)

        filePrefix = os.path.join(FILEDIR, 'data/path_test')
        if MPI.rank == 0 and os.path.isfile(f'{filePrefix}.h5'):
            os.remove(f'{filePrefix}.h5')

        with HDF5Handler(filePrefix) as hdf:
            sim.toDB(hdf, f'path')

            sim.newRun()

            sim.patch1.raft1.Count = 2

            path1_inj = range(0, 10000, 2000)
            path2_inj = range(800, 10000, 2000)

            tpnts = np.linspace(0, NTPNTS * DT, NTPNTS + 1)
            # DT is one millisecond, at 1 um/s a vesicle is expected to travel the 0.5um distance in 500ms.
            # So let's inject on every 100ms, alternating between path 1 and path 2 (by controlling the surface species)

            for i, t in enumerate(tpnts):
                if i in path1_inj:
                    v = sim.comp1.addVesicle(ves1)
                    v('surf').spec1.Count = 2
                    v.Pos = (0, 0, -2.2e-7)
                elif i in path2_inj:
                    v = sim.comp1.addVesicle(ves1)
                    v('surf').spec2.Count = 2
                    v.Pos = (0, 0, -2.2e-7)

                sim.run(t)

        if MPI.rank == 0:
            with HDF5Handler(filePrefix) as hdf:
                vesPos, = hdf['path'].results
                res = {}
                for t, posDct in zip(vesPos.time[0], vesPos.data[0, :, 0]):
                    for idx, pos in posDct.items():
                        times, posz = res.setdefault(idx, ([], []))
                        times.append(t)
                        posz.append(pos[2] / scale)

                analytical1 = np.arange(
                    -2.2e-7, 1.84e-7, speed1 * 1e-3) / scale  # distances travelled in 1ms
                analytical2 = np.arange(
                    -2.2e-7, 1.84e-7, speed2 * 1e-3) / scale  # distances travelled in 1ms

                lw = 3
                zshift = 0.25  # To start at zero
                labelled = False
                for ts in path1_inj:
                    if labelled:
                        plt.plot(tpnts[ts:ts + len(analytical1)],
                                 analytical1 + zshift,
                                 'k-',
                                 linewidth=lw)
                    else:
                        plt.plot(tpnts[ts:ts + len(analytical1)],
                                 analytical1 + zshift,
                                 'k-',
                                 linewidth=lw,
                                 label='expected')
                        labelled = True
                for ts in path2_inj:
                    plt.plot(tpnts[ts:ts + len(analytical2)],
                             analytical2 + zshift,
                             'k-',
                             linewidth=lw)

                label2 = False
                label1 = False
                for v in res.keys():
                    if v % 2:
                        if label1:
                            plt.plot(res[v][0],
                                     np.array(res[v][1]) + zshift,
                                     '--',
                                     linewidth=lw, color='deepskyblue')
                        else:
                            plt.plot(res[v][0],
                                     np.array(res[v][1]) + zshift,
                                     '--',
                                     linewidth=lw, color='deepskyblue',
                                     label='vesicle on path2')
                            label1 = True
                    else:
                        if label2:
                            plt.plot(res[v][0],
                                     np.array(res[v][1]) + zshift,
                                     '--',
                                     linewidth=lw, color='orange')
                        else:
                            plt.plot(res[v][0],
                                     np.array(res[v][1]) + zshift,
                                     '--',
                                     linewidth=lw, color='orange',
                                     label='vesicle on path1')
                            label2 = True

                plt.xlabel('Time (s)')
                plt.ylabel('z position ($\mu$m)')
                plt.legend()
                plt.ylim(0, 0.65)
                fig = plt.gcf()
                fig.set_size_inches(3.4, 3.4)
                fig.savefig(os.path.join(FILEDIR, "plots/path.pdf"), dpi=300, bbox_inches='tight')
                plt.close()

                for v in res.keys():
                    result = np.array(res[v][1]) + zshift
                    analy = (analytical2 if v % 2 else analytical1) + zshift
                    result, analy = map(np.array, zip(*zip(result, analy))) # Match lengths
                    self.assertTrue(np.allclose(analy, result, rtol=0.01, atol=0.05))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(VesiclePath))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
