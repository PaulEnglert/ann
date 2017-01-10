# -*- coding: utf-8 -*-

from .context import ann

from ann import core

import unittest


class AdvancedTestSuite(unittest.TestCase):
	"""Advanced test cases."""

	def test_perceptron_sn(self):
		data = [
			([0.5, 2], [1]),
			([0.6, 5], [1]),
			([-0.6, 0.1], [1]),
			([-0.8, 2], [1]),
			([-0.3, -1], [-1]),
			([-1, -0.1], [-1]),
			([3, -2.6], [-1]),
			([2.4, -0.34], [-1])
		]
		# setup network
		network = core.sl_network(2)
		network.build(1, 'step')

		assert len(network.neurons) == 1
		assert network.neurons[0].activation_type == 'step'

		# execute learning
		network.learn(data, 50)

		assert network.neurons[0].output is not None
		assert network.neurons[0].wsi is not None
		assert network.reached_zero_error

		# assert prediction
		assert network.classify([0.12, 0.9])[0] == 1


	def test_perceptron_mn(self):
		data = [
			([0.5, 2], [1, 1]),
			([0.6, 5], [1, 1]),
			([-0.6, -0.1], [-1, -1]),
			([-0.8, -2], [-1, -1]),
			([0.3, -1], [-1, 1]),
			([1, -0.1], [-1, 1]),
			([-3, 2.6], [1, -1]),
			([-2.4, 0.34], [1, -1])
		]
		# setup network
		network = core.sl_network(2)
		network.build(2, 'step')

		assert len(network.neurons) == 2
		assert network.neurons[0].activation_type == 'step'
		assert network.neurons[1].activation_type == 'step'

		# execute learning
		network.learn(data, 50)

		assert network.neurons[0].output is not None
		assert network.neurons[0].wsi is not None
		assert network.neurons[1].output is not None
		assert network.neurons[1].wsi is not None
		assert network.reached_zero_error

		# assert prediction
		output = network.classify([0.12, 0.9])
		assert output[0] == 1 and output[1] == 1


if __name__ == '__main__':
	unittest.main()