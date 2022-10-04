import contextlib
import os.path as osp
import unittest

if __name__ == '__main__':
    test_dir = osp.dirname(osp.abspath(__file__))
    loader = unittest.TestLoader()
    # Redirect stdout to avoid STEPS prints on module import
    with contextlib.redirect_stdout(None):
        top_dir = osp.join(test_dir, '..')
        suite = loader.discover(test_dir, pattern='test_*.py', top_level_dir=top_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)
