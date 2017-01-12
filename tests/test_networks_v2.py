# -*- coding: utf-8 -*-

from .context import ann
import numpy as np

from ann import core

import unittest


class NetworksTestSuiteV2(unittest.TestCase):
	"""Basic test cases."""

	def test_restricted_boltzmann_machine(self):
		num_inputs = 6
		num_hidden_units = 2
		learning_rate = 0.5
		trainX = np.asarray([
			[1,1,1,0,0,0],
			[1,0,1,0,0,0],
			[1,1,1,0,0,0],
			[0,0,1,1,1,0], 
			[0,0,1,1,0,0],
			[0,0,1,1,1,0],
		])
		trainY = np.asarray([1,1,1,0,0,0])

		network = core.v2RBMNetwork(num_inputs, num_hidden_units, learning_rate)

		network.train(500, trainX)

		network.label_units(trainX, trainY)
		network.print_labelling()

		prediction = network.predict(np.asarray([[0,0,0,1,1,0]]))
		network.print_prediction(prediction)

		n = 10
		dreamed = network.daydream(n)
		print('\nDaydreaming for '+str(n)+' gibbs steps:')
		print(dreamed)


	def test_stacked_rbm_dbm(self):
		num_inputs = 6
		num_outputs = 2
		num_layers = 2
		learning_rate = 0.5
		trainX = np.asarray([
			[1,1,1,0,0,0],
			[1,0,1,0,0,0],
			[1,1,1,0,0,0],
			[0,0,1,1,1,0], 
			[0,0,1,1,0,0],
			[0,0,1,1,1,0],
		])
		trainY = np.asarray([1,1,1,0,0,0])

		network = core.DBNetwork(num_inputs, num_outputs, num_layers, learning_rate)

		network.train(5, trainX)

		network.label_units(trainX, trainY)
		network.print_labelling()

		prediction = network.predict(np.asarray([[0,0,0,1,1,0]]))
		network.print_prediction(prediction)


if __name__ == '__main__':
	unittest.main()