from __future__ import absolute_import

import os.path as osp
import sys
import collections
import collections.abc
collections.Callable = collections.abc.Callable

import nose

from config import configuration




DEFAULT_TEST_SUITE = [
    'validation_rd_mpi/unbdiff2D.py',
    'validation_rd_mpi/unbdiff2D_linesource_ring.py',
    'validation_rd_mpi/unbdiff.py',
    'validation_rd_mpi/bounddiff.py',
    'validation_rd_mpi/csd_clamp.py',
    'validation_rd_mpi/masteq_diff.py',
    'validation_rd_mpi/kisilevich.py',
    'validation_efield_mpi/test_rallpack1_dist.py',
]


if __name__ == '__main__':
    test_dir = osp.dirname(osp.abspath(__file__))
    test_suite = sys.argv[1:] or DEFAULT_TEST_SUITE
    configuration.suffix = '_mpi'
    for suite in test_suite:
        nose.run(argv=['-s', '-v', osp.join(test_dir, suite)])
