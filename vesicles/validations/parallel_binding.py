########################################################################

# Test the binding reaction as a second-order reaction.

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

class VesicleBinding(unittest.TestCase):
    def test_binding(self):
        ########################################################################

        # Simulation parameters
        scale = 1e-6
        ENDT = 0.05
        DT = 0.01

        ########################################################################

        # Vesicle-related parameters
        ves_number = 200
        ves_diam = 5e-9
        spec1_number_perves = 1

        ########################################################################

        # Second order irreversible parameters
        KCST_soA2 = 2e6
        NITER_soA2 = 100

        AVOGADRO = 6.022e23

        ########################################################################

        model = Model()

        with model:
            vsys = VolumeSystem.Create()

            Ves1 = Vesicle.Create(ves_diam, 1e-12)
            Spec1 = Species.Create()

            Linkspec1 = LinkSpecies.Create()

            with vsys:
                vbind1 = VesicleBind.Create((Ves1, Ves1), (Spec1, Spec1),
                                            (Linkspec1, Linkspec1), 0, 1e-6, KCST_soA2,
                                            NO_EFFECT)

        ########################################################################

        mesh = TetMesh.LoadAbaqus(os.path.join(FILEDIR, 'meshes/sphere_0.5D_577tets.inp'), scale)

        with mesh:
            cyto = Compartment.Create(mesh.tets, vsys)
            memb = Patch.Create(mesh.surface, cyto)

        ########################################################################

        CONCA_soA2 = (ves_number * spec1_number_perves) / (AVOGADRO * cyto.Vol * 1e3)

        rng = RNG('mt19937', 512, 100)

        sim = Simulation('TetVesicle', model, mesh, rng, MPI.EF_NONE, check=False)

        rs = ResultSelector(sim)

        spec1_count = rs.cyto.Ves1('surf').Spec1.Count
        link1_count = rs.cyto.Ves1('surf').Linkspec1.Count

        sim.toSave(spec1_count, link1_count, dt=DT)

        filePrefix = os.path.join(FILEDIR, 'data/binding_test')
        if MPI.rank == 0 and os.path.isfile(f'{filePrefix}.h5'):
            os.remove(f'{filePrefix}.h5')

        with HDF5Handler(filePrefix) as hdf:
            sim.toDB(hdf, 'binding')

            for i in range(0, NITER_soA2):
                sim.newRun()

                if MPI.rank == 0:
                    print(i + 1, 'of', NITER_soA2)

                sim.cyto.Ves1.Count = ves_number

                sim.cyto.VESICLES(Ves1)('surf').Spec1.Count = spec1_number_perves

                sim.run(ENDT)

        if MPI.rank == 0:
            with HDF5Handler(filePrefix) as hdf:
                spec1_count, link1_count = hdf['binding'].results

                mean_res_soA2_spec1 = np.mean(spec1_count.data,
                                            axis=0) / (AVOGADRO * cyto.Vol * 1e3)
                mean_res_soA2_link = np.mean(link1_count.data,
                                            axis=0) / (AVOGADRO * cyto.Vol * 1e3)

                tpnts = spec1_count.time[0]
                invA = 1.0 / mean_res_soA2_spec1
                lineA = 1.0 / CONCA_soA2 + tpnts * 2 * KCST_soA2

                plt.plot(tpnts, lineA * 1e-6, 'k-', label='analytical', linewidth=3)
                plt.plot(tpnts, invA * 1e-6, 'c--', label='STEPS', linewidth=3)
                plt.xlabel('Time (s)')
                plt.ylabel('Inverse concentration (1/$\mu$M)')
                plt.legend()
                fig = plt.gcf()
                fig.set_size_inches(3.4, 3.4)
                fig.savefig(os.path.join(FILEDIR, 'plots/binding.pdf'), dpi=300, bbox_inches='tight')
                plt.close()

                self.assertTrue(np.allclose(invA.flatten(), lineA.flatten(), rtol=0.05))

########################################################################

def suite():
    all_tests = []
    all_tests.append(unittest.TestLoader().loadTestsFromTestCase(VesicleBinding))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
