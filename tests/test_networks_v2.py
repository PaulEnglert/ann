# -*- coding: utf-8 -*-

from .context import ann
import numpy as np

from ann import core, helpers

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
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33, train_size=0.67, random_state=42)

		learning_rate = 0.05
		network = core.DBNetwork(trainX.shape[1], len(np.unique(dataset.target)), 0, learning_rate, size_hidden_layers=300)

		epochs=5
		network.train(epochs, trainX, trainY, num_gibbs_sampling_steps=1)

		# network.label_units(trainX, trainY)
		network.print_labelling_probs()
		network.print_labelling()

		# predict all and calculate statistics
		print('\nStatistics')
		prediction = network.predict(testX)
		try:
			pred_labels = []
			for p in prediction:
				pred_labels.append(network.get_label(p))
			print(classification_report(testY, pred_labels))
		except:
			network.print_labelling_probs()
		print('\n')
		
		# for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
		# 	# classify the digit
		# 	pred = network.predict(np.atleast_2d(testX[i]))
		# 	# show the image and prediction
		# 	print "Actual digit is {0}, predicted {1}".format(testY[i], network.get_label(pred[0]))
		
			# image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
			# cv2.imshow("Digit", image)
			# cv2.waitKey(0)
	
	def test_mlp_v2(self):
		#np.seterr(all='raise')
		
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.033, train_size=0.067, random_state=42)

		batch_size = 100
		trainXbatches = helpers.util.create_minibatches(trainX, batch_size)
		trainYbatches = helpers.util.create_minibatches(trainY, batch_size)
		testXbatches = helpers.util.create_minibatches(testX, batch_size)
		testYbatches = helpers.util.create_minibatches(testY, batch_size)
		print trainXbatches[0].shape
		num_features = trainX.shape[1]
		layer_definition = [num_features,300,10]
		network = core.v2MLNetwork(layer_definition, batch_size)


		network.train(trainXbatches, trainYbatches, testXbatches, testYbatches, 50)

	# def test_mlp_v2(self):
		# trainX = np.asarray([ # boolean 'xor' function
		# 	[1, 1],
		# 	[1, 0],
		# 	[0, 0],
		# 	[0, 1]
		# ])
		# trainY = np.asarray([0,1,0,1])

		# learning_rate = 0.5
		# use_alpha=False

		# network = core.v2MLNetwork(learning_rate, use_alpha, debug=False)

		# network.construct([2,5,2], weights=None)
		# # network.construct([trainX.shape[1],6,len(np.unique(trainY))], weights=None)

		# epochs=1500
		# network.train(epochs, trainX, trainY)

		# # get a prediction
		# prediction = network.predict(trainX, probabilities=False)
		# print prediction
		# prediction = network.predict(trainX, probabilities=True)
		# print prediction

	# def test_mlp_v2_mnist(self):
		# np.seterr(all='raise')
		# print "(downloading data...)"
		# dataset = datasets.fetch_mldata("MNIST Original")
		# (trainX, testX, trainY, testY) = train_test_split(
		# 	dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.033, train_size=0.067, random_state=42)

		# learning_rate = 0.05
		# use_alpha=False

		# network = core.v2MLNetwork(learning_rate, use_alpha, debug=False)
		# 				#  features, hid1, hid2, out
		# network.construct([784, 100, 100, 10], weights=None)

		# epochs=50
		# network.train(epochs, trainX, trainY, batch_size=100)

		# # get a prediction
		# with open('/Users/paulenglert/Development/DataScience/ANN/tests/mlp.out', 'w') as file:
		# 	for y_, y in zip(network.predict(testX, probabilities=False), testY):
		# 		file.write(str(y_)+' (actual:'+str(y)+')\n')



if __name__ == '__main__':
	unittest.main()