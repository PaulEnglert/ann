# -*- coding: utf-8 -*-

import numpy as np


"""
Representation of a restricted boltzmann machine
"""
class RBMNetwork:

	def __init__(self, num_inputs, num_hidden_neurons, learning_rate, **kwargs):
		self.num_visible = num_inputs
		self.num_hidden = num_hidden_neurons
		self.learning_rate = learning_rate
		self.is_constructed = False
		self.is_trained = False
		self.log = kwargs.get('log', True)

	def construct(self):
		# build weight matrix
		self.weights = np.random.randn(self.num_visible, self.num_hidden)*0.1
		# add bias weights column
		self.weights = np.insert(self.weights, 0, 0, axis=1)
		# add bias row
		self.weights = np.insert(self.weights, 0, 0, axis=0)
		if self.log:
			self.print_weights()
		self.is_constructed = True


	def train(self, num_epochs, trainX, trainY = None, testX = None, testY = None, use_states=False, log_epochs=False):
		if not self.is_constructed:
			self.construct()

		# prepare data
		data = trainX
		data = np.insert(data, 0, 1, axis=1) # add column of bias states, all -> 1

		# run 
		for epoch in range(num_epochs):
			if self.log or log_epochs:
				print('Epoch '+str(epoch))

			# forward (positive) move: compute the states& probabilities of the hidden units, based on the real data
			hid_aes = np.dot(data, self.weights)
			hid_probs = self._logistic_function(hid_aes)
			hid_states  = hid_probs > np.random.randn(data.shape[0], self.num_hidden+1)
			# calculate association -> meaning calculate which units are on together and which are not (probability based, could also be states instead)
			if use_states:
				pos_association = np.dot(np.transpose(data.astype(np.bool)), hid_states)
			else:
				pos_association = np.dot(np.transpose(data.astype(np.bool)), hid_probs) # vis_probs is not known yet, we'll take the states, which are equal to the dataset plus the bias
			
			# backward (negative) move: compute the states & probabilities of the visible units based on the hidden states
			vis_aes = np.dot(hid_states, np.transpose(self.weights))
			vis_probs = self._logistic_function(vis_aes)
			vis_probs[:,0] = 1 # fix bias unit

			# forward (negative) move: recompute the states & probabilities of the hidden units, this time based on the expected outcome of the reverse step before
			hid_aes = np.dot(vis_probs, self.weights)
			hid_probs = self._logistic_function(hid_aes)
			if use_states:
				hid_states  = hid_probs > np.random.randn(data.shape[0], self.num_hidden+1)
				vis_states  = vis_probs > np.random.randn(data.shape[0], self.num_visible+1)
				neg_association = np.dot(np.transpose(vis_states), hid_states)
			else:
				neg_association = np.dot(np.transpose(vis_probs), hid_probs)

			# update weights
			self.weights = self.weights + self.learning_rate*((pos_association - neg_association) / data.shape[0])
			# measure
			if self.log:
				error = np.sum((data - vis_probs) ** 2)
				print('	100%:\n	error='+str(error)+'\n	...squared difference of data to expected visible state (probability)')
		
		if self.log:
			self.print_weights()
		self.is_trained = True


	def label_units(self, trainX, trainY):
		labels = np.unique(trainY)
		output = self.predict(trainX)
		# compute the average probabilities of a hidden unit for each class
		output = np.insert(output, 0, trainY, axis=1) # add labels in first column
		self.labelling=[]
		for l in labels:
			self.labelling.append([l]+[np.average(output[np.where(output[...,0]==l)][...,u]) for u in range(1,self.num_hidden+1)])
		return self.labelling

	def predict(self, data, return_states=False):
		if not self.is_trained:
			raise Exception('Network is not trained yet.')
		data = np.append(data, np.ones((data.shape[0], 1), dtype=data.dtype), axis=1) # add column of bias states, all -> 1
		vis_states = data.astype(bool)
		hid_aes = np.dot(vis_states, self.weights)
		hid_probs = self._logistic_function(hid_aes)
		hid_states  = hid_probs > np.random.randn(data.shape[0], self.num_hidden+1)
		if return_states:
			return hid_states[:,1:]
		else:
			return hid_probs[:,1:]

	def daydream(self, num_steps):
		if not self.is_trained:
			raise Exception('Network is not trained yet.')
		random_data = np.ones((num_steps, self.num_visible+1)) # create random data container
		random_data[0,1:] = np.random.rand(self.num_visible) # add random data to first row, the next rows will be populated in each iteration
		for step in range(num_steps):
			if step == num_steps-1:
				break
			# calculate hidden units states
			hid_aes = np.dot(random_data.astype(bool), self.weights)
			hid_probs = self._logistic_function(hid_aes)
			hid_probs[:,0] = 1 # fix bias unit
			hid_states  = hid_probs > np.random.randn(random_data.shape[0], self.num_hidden+1)
			# calculate visible units states
			vis_aes = np.dot(hid_states, np.transpose(self.weights))
			vis_probs = self._logistic_function(vis_aes)
			vis_probs[:,0] = 1 # fix bias unit
			vis_states  = vis_probs > np.random.randn(random_data.shape[0], self.num_visible+1)
			# set the newly calculated visible states as the next data row in the random_data
			random_data[step+1,:] = vis_states[step]
		return random_data[:,1:]

	# Utility functions

	def _logistic_function(self, x):
		return 1 / (1 + np.exp(-x))

	def print_weights(self):
		print('')
		header = '/;BIAS;'
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+(';' if i < self.num_hidden-1 else '')
		print(header)
		row_labels = ['BIAS']+['INP_'+str(i) for i in range(self.num_visible)]
		for row_label, row in zip(row_labels, self.weights):
			print '%s;%s' % (row_label, ';'.join('%03s' % i for i in row))

	def print_labelling(self):
		print('')
		header = 'label;'
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+(';' if i < self.num_hidden-1 else '')
		print(header)
		for row in self.labelling:
			print(';'.join(str(i) for i in row))

	def print_prediction(self, prediction):
		print('')
		header = ''
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+(';' if i < self.num_hidden-1 else '')
		print(header)
		for row in prediction:
			print(';'.join(str(i) for i in row))
