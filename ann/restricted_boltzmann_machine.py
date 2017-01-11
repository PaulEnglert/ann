# -*- coding: utf-8 -*-

from . import helpers
from helpers import util

from . import network

from random import random

"""
Representation of a neuron in the network 
"""
class RBMNeuron(network.Neuron):

	def __init__(self, num_inputs, learning_rate, activation_type, **kwargs):
		network.Neuron.__init__(self, num_inputs, learning_rate, activation_type, **kwargs)
		self.layer = kwargs.get('layer',0)
		self.state = 0

	def update_state(self):
		self.state = 0 if self.output < random() else 1 # output refers to the activation energy and is computed through the activation funciton on the weighted sum of inputs


"""
Representation of a restricted boltzmann machine
"""
class RBMNetwork(network.Network):

	def __init__(self, num_inputs, num_hidden_units, learning_rate):
		network.Network.__init__(self, num_inputs, learning_rate)
		self.num_hidden_units = num_hidden_units
		self.pos_association = {}
		self.neg_association = {}

	def build(self, activation_type='sigmoidal'):
		layers = []
		layers.insert(0, self.num_inputs) 		# add input layer
		layers.append(self.num_hidden_units)	# add output layer
		self.num_layers = len(layers)
		self.neurons = []
		util.log('Network topology:')
		for index, layer_count in enumerate(layers):
			for n in xrange(0,layer_count):
				neuron = RBMNeuron((self.num_inputs if index == 0 else layers[index-1]), self.learning_rate, activation_type, layer=index)
				self.neurons.append(neuron)
			util.log(str(index)+': '+';'.join([str(n.id)+' (' +str(len(n.input_weights))+')' for n in self.get_layer_neurons(index)]))
		util.log('Built network with '+str(self.num_layers) + ' layers.')


	def learn(self, data, num_iterations):
		for i in xrange(0,num_iterations):
			for features, target in data:
				# TODO for now assume we have only binary input as the network!!
				
				#1. set input layer states to feature vector
				input_neurons = self.get_layer_neurons(0)
				for f_i, feature in enumerate(features):
					input_neurons[f_i].output = feature
					input_neurons[f_i].update_state()

				#2. set hidden layer states
				states = [n.state for n in input_neurons]
				outputs = [n.output for n in input_neurons]
				for n_idx, neuron in enumerate(self.get_layer_neurons(1)):
					neuron.compute_output(states)
					neuron.update_state()
					#2.1 update pos_association
					for idx, output in enumerate(outputs): # for idx, state in enumerate(states):
						self.pos_association[str(idx)+';'+str(n_idx)]=output*neuron.output # self.pos_association[str(idx)+';'+str(n_idx)]=state*neuron.state
				for idx, n in enumerate(self.neurons): # bias
					self.pos_association[str(idx)+';b']=n.output

				#3. reconstruct the feature vector
				for n_idx, neuron in enumerate(input_neurons):
					weights = [n.input_weights[n_idx] for n in self.get_layer_neurons(1)]
					neuron.compute_wsi([n.state for n in self.get_layer_neurons(1)], weights, 0)
					neuron.compute_output(None, no_update_wsi=True)

				#4. update hidden units again and calculate neg_association
				states = [n.state for n in input_neurons]
				outputs = [n.output for n in input_neurons]
				for n_idx, neuron in enumerate(self.get_layer_neurons(1)):
					neuron.compute_output(states)
					neuron.update_state()
					#4.1 update neg_association
					for idx, output in enumerate(outputs): # for idx, state in enumerate(states):
						self.neg_association[str(idx)+';'+str(n_idx)]=output*neuron.output # self.neg_association[str(idx)+';'+str(n_idx)]=state*neuron.state
					for idx, n in enumerate(self.neurons): # bias
						self.neg_association[str(idx)+';b']=n.output
					#4.2 update weights
					for w_idx, weight in enumerate(neuron.input_weights):
						neuron.input_weights[w_idx] = neuron.input_weights[w_idx] + neuron.learning_rate*(self.pos_association[str(w_idx)+';'+str(n_idx)] - self.neg_association[str(w_idx)+';'+str(n_idx)])
				# bias
				for idx, n in enumerate(self.neurons):
					neuron.theta_weight = neuron.theta_weight + neuron.learning_rate*(self.pos_association[str(idx)+';b'] - self.neg_association[str(idx)+';b'])

		util.log('Weights of each neuron:')
		util.log(';'.join(['id','w_bias']+['w_'+str(i) for i in range(0, self.num_inputs)]))
		for n in self.neurons:
			util.log(';'.join([str(n.id), str(n.theta_weight)]+[str(w) for w in n.input_weights]))

	def classify(self, features):
		input_neurons = self.get_layer_neurons(0)
		for f_i, feature in enumerate(features):
			input_neurons[f_i].output = feature
			input_neurons[f_i].update_state()

		states = [n.state for n in input_neurons]
		for n_idx, neuron in enumerate(self.get_layer_neurons(1)):
			neuron.compute_output(states)
			neuron.update_state()
		return [n.output for n in self.get_layer_neurons(1)]

