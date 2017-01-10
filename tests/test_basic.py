# -*- coding: utf-8 -*-

from .context import ann

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_something(self):
        assert True


if __name__ == '__main__':
    unittest.main()