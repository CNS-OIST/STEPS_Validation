import contextlib
import os.path as osp
import sys
import unittest

if __name__ == '__main__':
    test_dir = osp.dirname(osp.abspath(__file__))
    loader = unittest.TestLoader()

    directories = []
    for fold in sys.argv[1:]:
        directories.append(osp.join(test_dir, fold))
    if len(directories) == 0:
        directories.append(test_dir)

    for start_dir in directories:
        print(f'Running validations from {start_dir}')

        with contextlib.redirect_stdout(None):
            top_dir = osp.join(test_dir, '..')
            suite = loader.discover(test_dir, pattern='test_*.py', top_level_dir=top_dir)

        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
