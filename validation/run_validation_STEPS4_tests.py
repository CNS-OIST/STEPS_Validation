from __future__ import absolute_import

import sys
import unittest

from config import configuration


DEFAULT_TEST_SUITE = [
    "validation_efield_STEPS4.rallpack1",
]


if __name__ == "__main__":
    test_suite = sys.argv[1:] or DEFAULT_TEST_SUITE
    configuration.suffix = "_STEPS4"
    for suite in test_suite:
        unittest.main(module=suite)
