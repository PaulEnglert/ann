# -*- coding: utf-8 -*-

from . import helpers
from helpers import util

from . import network

from random import random
from math import fabs


"""
Representation of a neuron in the network 
"""
class MLNeuron(network.Neuron):

	def __init__(self, num_inputs, learning_rate, use_alpha, activation_type, **kwargs): # kwargs={af_param:float, layer:int} for modifying parameter a of sigmoidal function, and adding a layer information
		network.Neuron.__init__(self, num_inputs, learning_rate, activation_type, **kwargs)
		self.delta = None
		self.last_input = None
		self.layer = kwargs.get('layer',0)
		self.use_alpha = use_alpha

	def compute_output(self, input_data):
		network.Neuron.compute_output(self, input_data)
		self.last_input = input_data

	# complex delta rule
	def update_weights(self):
		delta_w = -1*self.learning_rate*self.delta
		self.theta_weight = self.theta_weight + delta_w*1 + (random() if self.use_alpha else 0)*delta_w*1
		for index, x in enumerate(self.input_weights):
			self.input_weights[index] = self.input_weights[index] + (delta_w * self.last_input[index]) + (random() if self.use_alpha else 0)*delta_w*self.last_input[index]

	def update_dr_output_error(self, target):
		self.delta = (self.output - target) * self.output * (1 - self.output)
	
	def update_dr_hidden_error(self, ws_nextlayer):
		self.delta = (1 - self.output) * self.output * ws_nextlayer


"""
Representation of a multi layer network composed of neurons organized in several layers
"""
class MLNetwork(network.Network):
	def __init__(self, num_inputs, learning_rate, use_alpha):
		network.Network.__init__(self, num_inputs, learning_rate)
		self.use_alpha = use_alpha

	def build(self, layers, activation_type, **kwargs):# layers should be list [num_input_units, num_hiddenunits-layer1, num_hiddenunits-layer2[, ...], num_output_units]
		self.num_layers = len(layers)
		if self.num_layers > 2:
			util.log('WARNING: Using more than one hidden layer leads to the "vanishing gradient phenomenon", which will result in useless training!')
		self.neurons = []
		for l_index, count_in_layer in enumerate(layers):
			for n in xrange(0, count_in_layer):
				kwargs['layer'] = l_index
				n = MLNeuron((self.num_inputs if l_index == 0 else layers[l_index-1]), self.learning_rate, self.use_alpha, activation_type, **kwargs)
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
							neur.update_weights()
				
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

