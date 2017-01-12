# -*- coding: utf-8 -*-

from .context import ann

from ann import network

import unittest


class NeuronsTestSuiteV1(unittest.TestCase):
	"""Basic test cases."""

	def test_step_activation_function(self):
		assert network.Neuron.step_function(-0.02) == -1
		assert network.Neuron.step_function(0.12) == 1
	
	def test_sigmoidal_activation_function(self):
		a = 2
		assert network.Neuron.sigmoidal_function(0.5, a) > 0.7 and network.Neuron.sigmoidal_function(0.5, a) < 0.8
		assert network.Neuron.sigmoidal_function(-0.5, a) > 0.2 and network.Neuron.sigmoidal_function(-0.5, a) < 0.3
		assert network.Neuron.sigmoidal_function(5, a) > 0.99 and network.Neuron.sigmoidal_function(5, a) <= 1
		assert network.Neuron.sigmoidal_function(-5, a) < 0.01 and network.Neuron.sigmoidal_function(-5, a) >= 0

	def test_hyperbolic_activation_function(self):
		assert network.Neuron.hyperbolic_function(0.5) > 0.4 and network.Neuron.hyperbolic_function(0.5) < 0.5
		assert network.Neuron.hyperbolic_function(-0.5) > -0.5 and network.Neuron.hyperbolic_function(-0.5) < -0.4
		assert network.Neuron.hyperbolic_function(5) > 0.99 and network.Neuron.hyperbolic_function(5) <= 1
		assert network.Neuron.hyperbolic_function(-5) < -0.99 and network.Neuron.hyperbolic_function(-5) >= -1

	def test_gaussian_activation_function(self):
		assert network.Neuron.gaussian_function(0) == 1
		assert network.Neuron.gaussian_function(1) > 0.3 and network.Neuron.gaussian_function(1) < 0.4
		assert network.Neuron.gaussian_function(-1) == network.Neuron.gaussian_function(1)


if __name__ == '__main__':
	unittest.main()