# -*- coding: utf-8 -*-

from .context import ann
import numpy as np

from ann import core

import unittest

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import cv2


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

		network = core.v2RBMNetwork(num_inputs, num_hidden_units, learning_rate, debug=True)

		network.train(500, trainX, num_cd_steps=5, no_decay=True)

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

		network = core.DBNetwork(num_inputs, num_outputs, num_layers, learning_rate, size_hidden_layers=4)

		network.train(10, trainX)

		network.label_units(trainX, trainY)
		network.print_labelling()

		prediction = network.predict(np.asarray([[0,0,0,1,1,0]]))
		network.print_prediction(prediction)

	def test_stacked_rbm_dbm_mnist(self):
		np.seterr(all='raise')

		print "(downloading data...)"
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.033, train_size=0.067, random_state=42)

		learning_rate = 0.05
		network = core.DBNetwork(trainX.shape[1], len(np.unique(dataset.target)), 0, learning_rate, size_hidden_layers=300)

		network.train(15, trainX, trainY, num_gibbs_sampling_steps=1)

		# network.label_units(trainX, trainY)
		network.print_labelling_probs()
		network.print_labelling()

		# predict all and calculate statistics
		#prediction = network.predict(testX)
		# TODO transform prediction into usable format for classification_report()
		#print(classification_report(testY, prediction))
		
		for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
			# classify the digit
			pred = network.predict(np.atleast_2d(testX[i]))
			# show the image and prediction
			try:
				print "Actual digit is {0}, predicted {1}".format(testY[i], network.get_label(pred[0]))
			except:
				print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])

			# image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
			# cv2.imshow("Digit", image)
			# cv2.waitKey(0)



if __name__ == '__main__':
	unittest.main()