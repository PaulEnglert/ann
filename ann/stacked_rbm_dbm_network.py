# -*- coding: utf-8 -*-

import numpy as np

from .restricted_boltzmann_machine_v2 import RBMNetwork


"""
Representation of a deep belief network based on stacked, greedy, layer-by-layer trained rbms
"""
class DBNetwork:

	def __init__(self, num_inputs, num_outputs, num_layers, learning_rate, **kwargs):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.num_layers = num_layers
		self.learning_rate = learning_rate
		self.is_constructed = False
		self.is_labeled = False
		self.size_hidden_layers = kwargs.get('size_hidden_layers', num_inputs*2)
		self.debug = kwargs.get('debug', False)

	def construct(self):
		print('1. Constructing')
		self.machines = []
		# input layer
		self.machines.append(RBMNetwork(self.num_inputs, self.size_hidden_layers, self.learning_rate, log=False))
		print('	inp:'+str(self.num_inputs))
		print('	l0:'+str(self.size_hidden_layers))
		# n hidden layer
		for l in range(self.num_layers):
			self.machines.append(RBMNetwork(self.size_hidden_layers, self.size_hidden_layers, self.learning_rate, log=False))
			print('	l'+str(l+1)+':'+str(self.size_hidden_layers))
		# ouput layer
		self.machines.append(RBMNetwork(self.size_hidden_layers, self.num_outputs, self.learning_rate, log=False))
		print('	out:'+str(self.num_outputs))
		print('	--> number of machines to train: '+str(len(self.machines)))
		self.is_constructed = True

	def train(self, epochs, trainX, trainY = None, num_gibbs_sampling_steps=1):
		if not self.is_constructed:
			self.construct()

		# greedy, layer-by-layer training of the individual boltzmann machines
		data = trainX
		print('	Using '+str(trainX.shape[0])+' observations with '+str(trainX.shape[1])+' features')
		print('2. Starting Layer-Wise Training:')
		for i, m in enumerate(self.machines):
			print('	-------------- starting machine '+str(i))
			if not m.is_constructed:
				if i > 0 and m.num_visible == self.machines[i-1].num_hidden and m.num_hidden == self.machines[i-1].num_visible:
					m.construct(np.transpose(self.machines[i-1].weights))
					print('	Passed on weights of previous machine.')
				else:
					m.construct()
			m.train(epochs, data, use_states=False, log_epochs=True, num_cd_steps=num_gibbs_sampling_steps, exit_on_error_increase=False)
			data = m.predict(data, return_states=False)
			print('	-------------- finished')
		print('\n 	Finished Layer-Wise Training')

		if trainY is not None:
			print('3. Labelling units')
			self.label_units(trainX, trainY)

	def predict(self, input):
		data = input
		for i, m in enumerate(self.machines):
			if i == len(self.machines)-1:
				data = m.predict(data, return_states=False) # the last machine returns probabilities
			else:
				data = m.predict(data, return_states=True) # pass on states not probabilities
		return data

	def label_units(self, dataX, dataY):
		labels = np.unique(dataY)
		output = self.predict(dataX)
		# compute the average probabilities of a hidden unit for each class
		output = np.insert(output, 0, dataY, axis=1) # add labels in first column
		self.labelling_probs=[]
		for l in labels:
			group = output[np.where(output[:,0]==l)]
			self.labelling_probs.append([l]+[np.average(group[:,unit]) for unit in range(1,output.shape[1])])
		self.labelling_probs = np.asarray(self.labelling_probs)
		
		# assign units a label
		self.unit_labels = {}
		labelled = 0
		lps = self.labelling_probs.copy()
		ls = lps[:,0].copy()
		lps=np.delete(lps, 0, axis=1)
		while labelled < len(labels):
			i,j = np.unravel_index(np.argmax(lps), lps.shape)
			self.unit_labels[ls.item(i)] = ( j ,lps.item((i,j)) ) # add to dict label:(unit, prob)
			lps[i,:] = -1
			lps[:,j] = -1
			labelled = labelled + 1

		self.is_labeled = True
		return self.labelling_probs

	def get_label(self, probs):
		if not self.is_labeled:
			raise Exception('Network Output Units have not been labeled yet')
		u = np.argmax(probs, axis=0)
		for label, unit_data in self.unit_labels.iteritems():
			unit, p = unit_data
			if unit == u+1:
				return int(label)
		raise Exception('Winning unit has no label')

	# utilities
	def print_labelling(self):
		print('')
		header = 'label;(unit, avg. probability)'
		print(header)
		for key, value in self.unit_labels.iteritems():
			print(str(key)+';'+str(value))

	def print_labelling_probs(self):
		print('')
		header = 'label;'
		for i in range(self.num_outputs):
			header = header + 'HID_'+str(i)+(';' if i < self.num_outputs-1 else '')
		print(header)
		for row in self.labelling_probs:
			print(';'.join(str(i) for i in row))

	def print_prediction(self, prediction):
		print('')
		header = ''
		for i in range(self.num_outputs):
			header = header + 'OUT_'+str(i)+(';' if i < self.num_outputs-1 else '')
		print(header)
		for row in prediction:
			print(';'.join(str(i) for i in row))



