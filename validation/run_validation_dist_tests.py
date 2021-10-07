from __future__ import absolute_import

import os.path as osp
import sys

import nose

from config import configuration


DEFAULT_TEST_SUITE = [
    'validation_rd_dist/kisilevich.py',
    'validation_rd_dist/masteq_diff.py',
    'validation_rd_dist/unbdiff.py',
]


if __name__ == '__main__':
    test_dir = osp.dirname(osp.abspath(__file__))
    test_suite = sys.argv[1:] or DEFAULT_TEST_SUITE
    configuration.suffix = '_dist'
    for suite in test_suite:
        nose.run(argv=['-s', '-v', osp.join(test_dir, suite)])
