import os.path as path
import unittest
import steps.mpi

from . import rallpack1_dist_vertcond
from ..config import Configuration

configuration = Configuration(__file__)

# defaults
C={ 'meshdir': configuration.path('validation_efield/meshes'),
    'mesh': 'axon_cube_L1000um_D866nm_1978tets',
    'meshfmt': 'xml',
    'meshscale': 1,
    'datadir': 'validation_efield/data/rallpack1_correct',
    'v0data': 'v0',
    'v1data': 'vx',
    'seed': 7,
    'plot': True }

class TestRallpack1_vertcond(unittest.TestCase):

    def test_rallpack1_dist_vertcond(self):
        "Rallpack1 - Parallel solver TetOpSplit with PETSc E-Field solver"

        params = rallpack1_dist_vertcond.sim_parameters

        meshfile = path.join(C['meshdir'],C['mesh'])
        meshfmt = C['meshfmt']
        meshscale = C['meshscale']
        v0data = path.join(C['datadir'],C['v0data'])
        v1data = path.join(C['datadir'],C['v1data'])
        seed = C['seed']

        simdata, rms_err_0um, rms_err_1000um = rallpack1_dist_vertcond.run_comparison(configuration, seed, meshfile, meshfmt, meshscale, v0data, v1data)

        print("rms error at 0um = " + str(rms_err_0um))
        print("rms error at 1000um = " + str(rms_err_1000um))

        if (C['plot']) and steps.mpi.rank == 0:
            simdata*=1e3 # Convert to ms, mV
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.plot(simdata[0,:], simdata[2,:], 'k-' ,label = 'Rallpack1, 0um', linewidth=3)
            plt.plot(simdata[0,:], simdata[1,:], 'r--', label = 'STEPS setVertVolRes test, 0um', linewidth=3)
            plt.legend(loc='best')
            plt.ylabel('Potential (mV)')
            plt.subplot(212)
            plt.plot(simdata[0,:], simdata[4,:], 'k-' ,label = 'Rallpack1, 1000um', linewidth=3)
            plt.plot(simdata[0,:], simdata[3,:], 'r--', label = 'STEPS setVertVolRes test, 1000um', linewidth=3)
            plt.legend(loc='best')
            plt.ylabel('Potential (mV)')
            plt.xlabel('Time (ms)')
            plt.show(block=True)

        max_rms_err = 0.1 # any numerical issues and this will go to crazy number like 1e+85
        assert(rms_err_0um < max_rms_err)
        assert(rms_err_1000um < max_rms_err)

def suite():
    all_tests = []
    all_tests.append(unittest.makeSuite(TestRallpack1_vertcond, "test"))
    return unittest.TestSuite(all_tests)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
