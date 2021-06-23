import unittest

from tests.test_constrained_min import TestLogBarrierMethods
from tests.test_unconstrained_min import TestGradientDescentMethods

if __name__ == '__main__':
    test_classes_to_run = [TestLogBarrierMethods, TestGradientDescentMethods]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)