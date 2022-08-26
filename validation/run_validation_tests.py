import os.path as osp
import sys
import collections.abc
collections.Callable = collections.abc.Callable

import nose


DEFAULT_TEST_SUITE = [
    'validation_cp',
    'validation_efield',
    'validation_rd',
]


def main():
    test_dir = osp.dirname(osp.abspath(__file__))
    test_suite = sys.argv[1:] or DEFAULT_TEST_SUITE
    for suite in test_suite:
        nose.run(argv=[
            __file__,
            '-s',
            '--all-modules',
            '-v',
            osp.join(test_dir, suite)
        ])


if __name__ == '__main__':
    main()
