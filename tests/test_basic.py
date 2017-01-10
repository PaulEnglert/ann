# -*- coding: utf-8 -*-

from .context import ann

from ann import core

import unittest


class BasicTestSuite(unittest.TestCase):
	"""Basic test cases."""

	def test_step_activation_function(self):
		assert core.neuron.step_function(-0.02) == -1
		assert core.neuron.step_function(0.12) == 1
	
	def test_sigmoidal_activation_function(self):
		a = 2
		assert core.neuron.sigmoidal_function(0.5, a) > 0.7 and core.neuron.sigmoidal_function(0.5, a) < 0.8
		assert core.neuron.sigmoidal_function(-0.5, a) > 0.2 and core.neuron.sigmoidal_function(-0.5, a) < 0.3
		assert core.neuron.sigmoidal_function(5, a) > 0.99 and core.neuron.sigmoidal_function(5, a) <= 1
		assert core.neuron.sigmoidal_function(-5, a) < 0.01 and core.neuron.sigmoidal_function(-5, a) >= 0

	def test_hyperbolic_activation_function(self):
		assert core.neuron.hyperbolic_function(0.5) > 0.4 and core.neuron.hyperbolic_function(0.5) < 0.5
		assert core.neuron.hyperbolic_function(-0.5) > -0.5 and core.neuron.hyperbolic_function(-0.5) < -0.4
		assert core.neuron.hyperbolic_function(5) > 0.99 and core.neuron.hyperbolic_function(5) <= 1
		assert core.neuron.hyperbolic_function(-5) < -0.99 and core.neuron.hyperbolic_function(-5) >= -1

	def test_gaussian_activation_function(self):
		assert core.neuron.gaussian_function(0) == 1
		assert core.neuron.gaussian_function(1) > 0.3 and core.neuron.gaussian_function(1) < 0.4
		assert core.neuron.gaussian_function(-1) == core.neuron.gaussian_function(1)


if __name__ == '__main__':
	unittest.main()