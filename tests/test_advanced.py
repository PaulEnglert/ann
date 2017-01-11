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
		network = core.SLNetwork(2, 0.5)
		network.build(1, 'step')

		assert len(network.neurons) == 1
		assert network.neurons[0].activation_type == 'step'

		# execute learning
		network.learn(data, 50)

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
		network = core.SLNetwork(2, 0.5)
		network.build(2, 'step')

		assert len(network.neurons) == 2
		assert network.neurons[0].activation_type == 'step'
		assert network.neurons[1].activation_type == 'step'

		# execute learning
		network.learn(data, 50)

		assert network.reached_zero_error

		# assert prediction
		output = network.classify([0.12, 0.9])
		assert output[0] == 1 and output[1] == 1

	def test_multi_layer(self):
		data = [ # boolean 'xor' function
			([1, 1], [0]),
			([1, 0], [1]),
			([0, 0], [0]),
			([0, 1], [1])
		]
		# setup network
		network = core.MLNetwork(2, 0.5, False)
		layers = [5,1]#[2,2,1]
		n_neruons = sum(layers)
		network.build(layers, 'sigmoidal')

		assert network.num_layers == len(layers)
		assert len(network.neurons) == n_neruons

		# execute learning
		network.learn(data, 1500)


	def test_restricted_boltzmann_machine(self):
		num_inputs = 6
		num_hidden_units = 2
		learning_rate = 0.5
		data = [
			([1,1,1,0,0,0],1), # data clustered into 1 and 0 -> that's what the rbm should find out
			([1,0,1,0,0,0],1),
			([1,1,1,0,0,0],1),
			([0,0,1,1,1,0],0), 
			([0,0,1,1,0,0],0),
			([0,0,1,1,1,0],0)
		]
		# setup network
		network = core.RBMNetwork(num_inputs, num_hidden_units, learning_rate)
		n_neruons = num_inputs+num_hidden_units
		network.build(activation_type='sigmoidal')

		assert network.num_layers == 2
		assert len(network.neurons) == n_neruons

		# execute learning
		network.learn(data, 1500)

		print('\nClassifying known [0,0,1,1,1,0] (to determine which hidden neuron belongs to which class:')
		outputs = network.classify([0,0,1,1,1,0])
		print(';'.join([str(o) for o in outputs]))
		print('\nClassifying new [0,0,0,1,1,0]:')
		outputs = network.classify([0,0,0,1,1,0])
		print(';'.join([str(o) for o in outputs]))

	def test_restricted_boltzmann_machine_nb(self):
		num_inputs = 6
		num_hidden_units = 2
		learning_rate = 0.5
		data = [
			([0.89,0.76,0.8,0,0.124,0.32],1), # data clustered into 1 and 0 -> that's what the rbm should find out
			([0.9,0.2,0.78,0.1,0.3,0.04],1),
			([0.93,0.73,0.76,0.012,0.1,0.3],1),
			([0.23,0.11,0.86,0.95,0.995,0.21],0), 
			([0.2,0.04,0.954,0.756,0.122,0.44],0),
			([0.3,0.21,0.89,0.865,0.99,0.12],0)
		]
		# setup network
		network = core.RBMNetwork(num_inputs, num_hidden_units, learning_rate)
		n_neruons = num_inputs+num_hidden_units
		network.build(activation_type='sigmoidal')

		assert network.num_layers == 2
		assert len(network.neurons) == n_neruons

		# execute learning
		network.learn(data, 1500)

		print('\nClassifying known [0.23,0.11,0.86,0.95,0.995,0.21] (to determine which hidden neuron belongs to which class:')
		outputs = network.classify([0.23,0.11,0.86,0.95,0.995,0.21])
		print(';'.join([str(o) for o in outputs]))
		print('\nClassifying new [0.13,0.32,0.22,0.94,0.72,0.04]:')
		outputs = network.classify([0.13,0.32,0.22,0.94,0.72,0.04])
		print(';'.join([str(o) for o in outputs]))

if __name__ == '__main__':
	unittest.main()