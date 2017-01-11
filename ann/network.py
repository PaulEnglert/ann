# -*- coding: utf-8 -*-

from random import random
from math import exp, pow


"""
Abstract Representation of a neuron in the network 
"""
class Neuron:
	count = 1
	ACTIVATION_TYPES = ['step', 'sigmoidal', 'hyperbolic', 'gaussian']

	def __init__(self, num_inputs, learning_rate, activation_type, **kwargs):
		self.id = Neuron.count
		Neuron.count = Neuron.count + 1

		self.learning_rate = learning_rate
		self.output = None
		self.wsi = None

		self.input_weights = []
		for x in xrange(0, num_inputs):
			self.input_weights.append(random())
		self.theta_weight = random()

		self.activation_type = activation_type
		if activation_type not in Neuron.ACTIVATION_TYPES:
			raise Exception('Activation Function Unknown.')
		self.af_param = kwargs.get('af_param',1)

	def compute_output(self, input_data):
		"""
		Compute the output value of the neuron 
		"""
		# compute weighted sum of inputs
		self.compute_wsi(input_data)
		# compute output based on initialization
		if self.activation_type == 'step':
			self.output = Neuron.step_function(self.wsi)
		elif self.activation_type == 'sigmoidal':
			self.output = Neuron.sigmoidal_function(self.wsi, self.af_param)
		elif self.activation_type == 'hyperbolic':
			self.output = Neuron.hyperbolic_function(self.wsi)
		elif self.activation_type == 'gaussian':
			self.output = Neuron.gaussian_function(self.wsi)

	def compute_wsi(self, input_data):
		"""
		Compute the weighted sum of input vectors
		"""
		self.wsi = self.theta_weight
		for index, value in enumerate(input_data):
			self.wsi = self.wsi + ( value * self.input_weights[index] )

	def update_weights(self):
		"""
		Update weights based on last input/output and learning rate
		"""
		pass


	@staticmethod
	def step_function(value):
		if value > 0:
			return 1
		else: 
			return -1

	@staticmethod
	def sigmoidal_function(value, a):
		return 1 / ( 1 + exp( -1 * a * value) )

	@staticmethod
	def hyperbolic_function(value):
		return ( exp( value ) - exp( -1 * value ) ) / ( exp( value ) + exp( -1 * value ) )

	@staticmethod
	def gaussian_function(value):
		return exp( -1 * pow( value, 2 ) )


class Network:
	def __init__(self, num_inputs, learning_rate):
		self.num_inputs = num_inputs
		self.learning_rate = learning_rate
		self.neurons = []
	
	def get_neuron(self, id):
		for n in self.neurons:
			if n.id == id: 
				return n
		return None

	def get_layer_neurons(self, layer):
		ns = []
		for n in self.neurons:
			if n.layer == layer: 
				ns.append(n)
		return ns

	def is_output_neuron(self, id):
		ons = self.get_layer_neurons(self.num_layers-1)
		for n in ons:
			if n.id == id:
				return True
		return False

	def build(self):
		pass

	def learn(self, data, num_iterations):
		pass

	def classify(self, features):
		pass
