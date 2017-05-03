import nose
import os
import sys

if __name__ == '__main__':
    # It's expected that we are being run from the test subdirectory,
    # and so all the test modules and the mesh directory is under
    # 'validation/'.

    nose.run(argv=['-s', '-v', 'validation_rd_mpi/unbdiff2D.py'])
    nose.run(argv=['-s', '-v', 'validation_rd_mpi/unbdiff2D_linesource_ring.py'])
    nose.run(argv=['-s', '-v', 'validation_rd_mpi/unbdiff.py'])
    nose.run(argv=['-s', '-v', 'validation_rd_mpi/bounddiff.py'])
    nose.run(argv=['-s', '-v', 'validation_rd_mpi/csd_clamp.py'])
    nose.run(argv=['-s', '-v', 'validation_rd_mpi/kisilevich.py'])
    nose.run(argv=['-s', '-v', 'validation_rd_mpi/masteq_diff.py'])
