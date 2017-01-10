# -*- coding: utf-8 -*-

from .context import ann

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_function(self):
        # execute and assert functions of module
        assert True


if __name__ == '__main__':
    unittest.main()