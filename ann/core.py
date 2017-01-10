# -*- coding: utf-8 -*-

from random import random
from math import exp, pow

"""
Representation of a neuron in the network 
"""
class neuron:

	ACTIVATION_TYPES = ['step', 'sigmoidal', 'hyperbolic', 'gaussian']

	def __init__(self, num_inputs, activation_type, **kwargs):
		self.output = None
		self.wsi = None
		# init weights
		self.input_weights = []
		for x in xrange(0, num_inputs):
			self.input_weights.append(random())
		self.theta_weight = random()
		self.activation_type = activation_type
		if activation_type not in neuron.ACTIVATION_TYPES:
			raise Exception('Activation Function Unknown.')
		if activation_type == 'sigmoidal':
			if kwargs is None or kwargs.get('af_param', None) is None:
				raise Exception('Sigmoidal Activation Function requires the kwarg \'af_param\'.')
			else:
				self.af_param = kwargs['af_param']

	def compute_output(self, input_data):
		"""
		Compute the output value of the neuron 
		"""
		# compute weighted sum of inputs
		self.compute_wsi(input_data)
		# compute output based on initialization
		if self.activation_type == 'step':
			self.output = neuron.step_function(self.wsi)
		elif self.activation_type == 'sigmoidal':
			self.output = neuron.sigmoidal_function(self.wsi, self.af_param)
		elif self.activation_type == 'hyperbolic':
			self.output = neuron.hyperbolic_function(self.wsi)
		elif self.activation_type == 'gaussian':
			self.output = neuron.gaussian_function(self.wsi)

	def compute_wsi(self, input_data):
		"""
		Compute the weighted sum of input vectors
		"""
		self.wsi = self.theta_weight
		input_data = input_data
		for index, value in enumerate(input_data):
			self.wsi = self.wsi + ( value * self.input_weights[index] )

	def update_weights(self, learning_rate, input_values, positive_diff):
		if positive_diff:
			self.theta_weight = self.theta_weight - learning_rate
			for index, x in enumerate(input_values):
				self.input_weights[index] = self.input_weights[index] - learning_rate*x
		else:
			self.theta_weight = self.theta_weight + learning_rate
			for index, x in enumerate(input_values):
				self.input_weights[index] = self.input_weights[index] + learning_rate*x

	@staticmethod
	def step_function(value):
		if value > 0:
			return 1
		else: 
			return -1

	@staticmethod
	def sigmoidal_function(value, a):
		return 1 / ( 1 + exp( -1 * a  * value) )

	@staticmethod
	def hyperbolic_function(value):
		return ( exp( value ) - exp( -1 * value ) ) / ( exp( value ) + exp( -1 * value ) )

	@staticmethod
	def gaussian_function(value):
		return exp( -1 * pow( value, 2 ) )

"""
Representation of a single layer network composed of neurons
"""
class sl_network:
	LEARNING_RATE = 0.5

	def __init__(self, num_inputs):
		self.num_inputs = num_inputs
		self.reached_zero_error = False

	def build(self, num_neurons, activation_type, **kwargs):
		self.neurons = []
		for i in xrange(0, num_neurons):
			self.neurons.append(neuron(self.num_inputs, activation_type, **kwargs))

	def learn(self, data, num_iterations):
		for i in xrange(0, num_iterations):
			sl_network.log('Iteration: '+str(i))
			sl_network.log('id;features[];target;output')

			zero_error = True
			for features, target in data:
				for index, neur in enumerate(self.neurons):
					neur.compute_output(features)
					if neur.output != target:
						neur.update_weights(sl_network.LEARNING_RATE, features, ( neur.output > target ))
						zero_error = False
					# log data and output
					sl_network.log_learning_step_data(index, features, target, neur.output)
			# log weights
			sl_network.log('id;theta;weights')
			for index, neur in enumerate(self.neurons):
				sl_network.log_learning_step_weights(index, neur.theta_weight, neur.input_weights)

			if zero_error:
				self.reached_zero_error = True
				break


	@staticmethod
	def log_learning_step_data(id, features, target, output):
		out = [id,]+features+[target,]+[output,]
		print(';'.join([str(item) for item in out]))
	
	@staticmethod
	def log_learning_step_weights(id, theta, weights):
		out = [id, theta]+weights
		print(';'.join([str(item) for item in out]))
	
	@staticmethod
	def log(line):
		print(str(line))




