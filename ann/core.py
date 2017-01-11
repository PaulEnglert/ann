# -*- coding: utf-8 -*-

import logging
import sys
from random import random
from math import exp, pow, fabs

# setup logging
log = logging.getLogger('ann.default')
out_hdlr = logging.StreamHandler(sys.stdout)
#out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setFormatter(logging.Formatter('%(message)s'))
out_hdlr.setLevel(logging.INFO)
log.addHandler(out_hdlr)
log.setLevel(logging.INFO)


class util:
	@staticmethod
	def log_learning_step_data(id, features, target, output):
		out = [id,]+features+[target,]+[output,]
		log.info(';'.join([str(item) for item in out]))
	
	@staticmethod
	def log_learning_step_weights(id, theta, weights):
		out = [id, theta]+weights
		log.info(';'.join([str(item) for item in out]))
	
	@staticmethod
	def log(line):
		log.info(str(line))


"""
Representation of a neuron in the network 
"""
class neuron:
	count = 1
	ACTIVATION_TYPES = ['step', 'sigmoidal', 'hyperbolic', 'gaussian']

	def __init__(self, num_inputs, activation_type, **kwargs): # kwargs={af_param:float, layer:int} for modifying parameter a of sigmoidal function, and adding a layer information
		self.layer = kwargs.get('layer',0)
		self.id = neuron.count
		neuron.count = neuron.count + 1
		self.output = None
		self.wsi = None
		self.delta = None
		self.last_input = None
		# init weights
		self.input_weights = []
		for x in xrange(0, num_inputs):
			self.input_weights.append(random())
		self.theta_weight = random()
		self.activation_type = activation_type
		if activation_type not in neuron.ACTIVATION_TYPES:
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
			self.output = neuron.step_function(self.wsi)
		elif self.activation_type == 'sigmoidal':
			self.output = neuron.sigmoidal_function(self.wsi, self.af_param)
		elif self.activation_type == 'hyperbolic':
			self.output = neuron.hyperbolic_function(self.wsi)
		elif self.activation_type == 'gaussian':
			self.output = neuron.gaussian_function(self.wsi)
		self.last_input = input_data

	def compute_wsi(self, input_data):
		"""
		Compute the weighted sum of input vectors
		"""
		self.wsi = self.theta_weight
		for index, value in enumerate(input_data):
			self.wsi = self.wsi + ( value * self.input_weights[index] )

	# simple perceptron/single layer update rule
	def update_weights(self, learning_rate, input_values, positive_diff):
		if positive_diff:
			self.theta_weight = self.theta_weight - learning_rate
			for index, x in enumerate(input_values):
				self.input_weights[index] = self.input_weights[index] - learning_rate*x
		else:
			self.theta_weight = self.theta_weight + learning_rate
			for index, x in enumerate(input_values):
				self.input_weights[index] = self.input_weights[index] + learning_rate*x

	# complex delta rule
	def update_weights_delta_rule(self, learning_rate, use_alpha=False):
		delta_w = -1*learning_rate*self.delta
		self.theta_weight = self.theta_weight + delta_w*1 + (random() if use_alpha else 0)*delta_w*1
		for index, x in enumerate(self.input_weights):
			self.input_weights[index] = self.input_weights[index] + (delta_w * self.last_input[index]) + (random() if use_alpha else 0)*delta_w*self.last_input[index]

	def update_dr_output_error(self, target):
		self.delta = (self.output - target) * self.output * (1 - self.output)
	
	def update_dr_hidden_error(self, ws_nextlayer):
		self.delta = (1 - self.output) * self.output * ws_nextlayer

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

"""
Representation of a multi layer network composed of neurons organized in several layers
"""
class ml_network:
	LEARNING_RATE = 0.5
	USE_ALPHA = False

	def __init__(self, num_inputs):
		self.num_inputs = num_inputs

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


	def build(self, layers, activation_type, **kwargs):# layers should be list [num_input_units, num_hiddenunits-layer1, num_hiddenunits-layer2[, ...], num_output_units]
		self.num_layers = len(layers)
		if self.num_layers > 2:
			util.log('WARNING: Using more than one hidden layer leads to the "vanishing gradient phenomenon", which will result in useless training!')
		self.neurons = []
		for l_index, count_in_layer in enumerate(layers):
			for n in xrange(0, count_in_layer):
				kwargs['layer'] = l_index
				n = neuron((self.num_inputs if l_index == 0 else layers[l_index-1]), activation_type, **kwargs)
				self.neurons.append(n)
			util.log(str(l_index)+': '+';'.join([str(n.id)+' (' +str(len(n.input_weights))+')' for n in self.get_layer_neurons(l_index)]))
		util.log('Built network with '+str(self.num_layers) + ' layers.')

	def learn(self, data, num_iterations):
		util.log('Learning for '+str(num_iterations) + ' iterations.')
		for i in xrange(0, num_iterations):
			util.log('Iteration: '+str(i))
			util.log('id;features[];target;output')

			for features, target in data:
				# calculate all outputs
				predicted = self.classify(features)
				
				if not self.is_satisfactory(target, predicted):				
					for l_index in reversed(range(0, self.num_layers)):
						# update deltas
						for n_index, neur in enumerate(self.get_layer_neurons(l_index)):
							if l_index == self.num_layers-1:
								neur.update_dr_output_error(target[n_index])
							else:
								ws_nextlayer = 0
								for n in self.get_layer_neurons(l_index+1):
									ws_nextlayer = ws_nextlayer + n.delta*n.input_weights[n_index]
								neur.update_dr_hidden_error(ws_nextlayer)
							neur.update_weights_delta_rule(ml_network.LEARNING_RATE, ml_network.USE_ALPHA)
				
				for index, neur in enumerate(self.neurons):
					if self.is_output_neuron(neur.id):
						util.log_learning_step_data(neur.id, features, ';'.join([str(t) for t in target]), neur.output)

	def classify(self, features):
		outputs = []
		for l_index in xrange(0, self.num_layers):
			next_outputs = []
			for neur in self.get_layer_neurons(l_index):
				neur.compute_output((features if l_index == 0 else outputs))
				next_outputs.append(neur.output)
			outputs = next_outputs
		return outputs

	def is_satisfactory(self, target, predicted):
		for index, t in enumerate(target):
			if fabs(predicted[index] - t) > 0.05:
				return False
		return True

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
						neur.update_weights(sl_network.LEARNING_RATE, features, ( neur.output > target[index] ))
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




