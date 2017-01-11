# -*- coding: utf-8 -*-

from . import helpers
from helpers import util

from . import network

from random import random
from math import exp, pow

"""
Representation of a neuron in the network 
"""
class SLNeuron(network.Neuron):

	def __init__(self, num_inputs, learning_rate, activation_type, **kwargs):
		network.Neuron.__init__(self, num_inputs, learning_rate, activation_type, **kwargs)

	# simple perceptron/single layer update rule
	def update_weights(self, input_values, positive_diff):
		if positive_diff:
			self.theta_weight = self.theta_weight - self.learning_rate
			for index, x in enumerate(input_values):
				self.input_weights[index] = self.input_weights[index] - self.learning_rate*x
		else:
			self.theta_weight = self.theta_weight + self.learning_rate
			for index, x in enumerate(input_values):
				self.input_weights[index] = self.input_weights[index] + self.learning_rate*x

"""
Representation of a single layer network composed of neurons
"""
class SLNetwork(network.Network):

	def __init__(self, num_inputs, learning_rate):
		network.Network.__init__(self, num_inputs, learning_rate)
		self.reached_zero_error = False

	def build(self, num_neurons, activation_type, **kwargs):
		self.neurons = []
		for i in xrange(0, num_neurons):
			self.neurons.append(SLNeuron(self.num_inputs, self.learning_rate, activation_type, **kwargs))
		util.log('Built network with '+str(num_neurons) + ' neurons.')

	def learn(self, data, num_iterations):
		util.log('Learning for '+str(num_iterations) + ' iterations.')
		for i in xrange(0, num_iterations):
			util.log('Iteration: '+str(i))
			util.log('id;features[];target;output')

			zero_error = True
			for features, target in data:
				for index, neur in enumerate(self.neurons):
					neur.compute_output(features)
					if neur.output != target[index]:
						neur.update_weights(features, ( neur.output > target[index] ))
						zero_error = False
					# log data and output
					util.log_learning_step_data(index, features, target[index], neur.output)
			# log weights
			util.log('id;theta;weights')
			for index, neur in enumerate(self.neurons):
				util.log_learning_step_weights(index, neur.theta_weight, neur.input_weights)

			if zero_error:
				self.reached_zero_error = True
				util.log('Stopping iteration because zero error has been reached.')
				break

	def classify(self, features):
		out = []
		for neur in self.neurons:
			neur.compute_output(features)
			out.append(neur.output)
		return out
