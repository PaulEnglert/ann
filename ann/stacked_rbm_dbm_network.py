# -*- coding: utf-8 -*-

import numpy as np

from .restricted_boltzmann_machine_v2 import RBMNetwork


"""
Representation of a deep belief network based on stacked, greedy, layer-by-layer trained rbms
"""
class DBNetwork:

	def __init__(self, num_inputs, num_outputs, num_layers, learning_rate):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.num_layers = num_layers
		self.learning_rate = learning_rate
		self.is_constructed = False

	def construct(self):
		self.machines = []
		# input layer
		self.machines.append(RBMNetwork(self.num_inputs, self.num_inputs*2, self.learning_rate))
		# n hidden layer
		for l in range(self.num_layers):
			self.machines.append(RBMNetwork(self.num_inputs*2, self.num_inputs*2, self.learning_rate))
		# ouput layer
		self.machines.append(RBMNetwork(self.num_inputs*2, self.num_outputs, self.learning_rate))
		for m in self.machines:
			m.construct()
		self.is_constructed = True

	def train(self, epochs, trainX):
		if not self.is_constructed:
			self.construct()

		# greedy, layer-by-layer training of the individual boltzmann machines
		data = trainX
		print('Starting Layer-Wise Training:')
		for i, m in enumerate(self.machines):
			m.train(epochs, data)
			data = m.predict(data)
			print('--------------machine '+str(i)+' finished training.')
		print('\nFinished Layer-Wise Training')

	def predict(self, input):
		data = input
		for i, m in enumerate(self.machines):
			data = m.predict(data)
		return data

	def label_units(self, dataX, dataY):
		labels = np.unique(dataY)
		output = self.predict(dataX)
		# compute the average probabilities of a hidden unit for each class
		output = np.insert(output, 0, dataY, axis=1) # add labels in first column
		self.labelling=[]
		for l in labels:
			self.labelling.append([l]+[np.average(output[np.where(output[...,0]==l)][...,u]) for u in range(1,self.num_outputs+1)])
		return self.labelling

	# utilities

	def print_labelling(self):
		print('')
		header = 'label;'
		for i in range(self.num_outputs):
			header = header + 'HID_'+str(i)+(';' if i < self.num_outputs-1 else '')
		print(header)
		for row in self.labelling:
			print(';'.join(str(i) for i in row))

	def print_prediction(self, prediction):
		print('')
		header = ''
		for i in range(self.num_outputs):
			header = header + 'OUT_'+str(i)+(';' if i < self.num_outputs-1 else '')
		print(header)
		for row in prediction:
			print(';'.join(str(i) for i in row))



