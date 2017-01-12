# -*- coding: utf-8 -*-

import numpy as np


"""
Representation of a restricted boltzmann machine
"""
class RBMNetwork:

	def __init__(self, num_inputs, num_hidden_neurons, learning_rate):
		self.num_visible = num_inputs
		self.num_hidden = num_hidden_neurons
		self.learning_rate = learning_rate
		self.is_constructed = False
		self.is_trained = False

	def construct(self):
		# build weight matrix
		self.weights = np.random.randn(self.num_visible, self.num_hidden)*0.1
		# add bias weights column
		self.weights = np.append(self.weights, np.zeros((self.weights.shape[0], 1), dtype=self.weights.dtype), axis=1)
		# add bias row
		self.weights = np.append(self.weights, np.zeros((1, self.weights.shape[1]), dtype=self.weights.dtype), axis=0)
		self.print_weights()
		self.is_constructed = True


	def train(self, num_epochs, trainX, trainY = None, testX = None, testY = None):
		if not self.is_constructed:
			self.construct()

		# prepare data
		self.data = trainX
		self.data = np.append(self.data, np.ones((self.data.shape[0], 1), dtype=self.data.dtype), axis=1) # add column of bias states, all -> 1

		# run 
		for epoch in range(num_epochs):
			print('Starting Epoch '+str(epoch))
			# forward (positive) move: compute the states& probabilities of the hidden units, based on the real data
			self.vis_states = self.data.astype(bool)
			self.compute_hid_states()
			# calculate association -> meaning calculate which units are on together and which are not (probability based, could also be states instead)
			self.pos_association = np.dot(np.transpose(self.vis_states), self.hid_probs) # vis_probs is not known yet, we'll take the states, which are equal to the dataset plus the bias
			# backward (negative) move: compute the states & probabilities of the visible units based on the hidden states
			self.compute_vis_states()
			# forward (negative) move: recompute the states & probabilities of the hidden units, this time based on the expected outcome of the reverse step before
			self.compute_hid_states(use_probs = True)
			self.neg_association = np.dot(np.transpose(self.vis_probs), self.hid_probs)
			# update weights
			self.update_weights()
			# measure
			error = np.sum((self.data - self.vis_probs) ** 2)
			print('	100%:\n	error='+str(error)+'\n	...squared difference of data to expected visible state (probability)')
		self.print_weights()
		self.is_trained = True

	def compute_hid_states(self, use_probs = False):
		if use_probs:
			self.hid_aes = np.dot(self.vis_probs, self.weights)
		else:
			self.hid_aes = np.dot(self.vis_states, self.weights)
		self.hid_probs = self._logistic_function(self.hid_aes)
		self.hid_states  = self.hid_probs > np.random.randn(self.data.shape[0], self.num_hidden+1)
	
	def compute_vis_states(self, use_probs = False):
		if use_probs:
			self.vis_aes = np.dot(self.hid_probs, np.transpose(self.weights))
		else:
			self.vis_aes = np.dot(self.hid_states, np.transpose(self.weights))
		self.vis_probs = self._logistic_function(self.vis_aes)
		# fix bias probability
		self.vis_probs[:,-1] = 1
		self.vis_states  = self.vis_probs > np.random.randn(self.data.shape[0], self.num_visible+1)


	def update_weights(self):
		self.weights = self.weights + self.learning_rate*(self.pos_association - self.neg_association)


	def label_units(self, trainX, trainY):
		labels = np.unique(trainY)
		output = self.predict(trainX)
		# compute the average probabilities of a hidden unit for each class
		output = np.insert(output, 0, trainY, axis=1) # add labels in first column
		self.labelling=[]
		for l in labels:
			self.labelling.append([l]+[np.average(output[np.where(output[...,0]==l)][...,u]) for u in range(1,self.num_hidden+1)])
		return self.labelling

	def predict(self, data):
		if not self.is_trained:
			raise Exception('Network is not trained yet.')
		data = np.append(data, np.ones((data.shape[0], 1), dtype=data.dtype), axis=1) # add column of bias states, all -> 1
		self.vis_states = data.astype(bool)
		self.compute_hid_states()
		return self.hid_probs[:,:-1]

	def daydream(self, num_steps):
		if not self.is_trained:
			raise Exception('Network is not trained yet.')
		random_data = np.ones((num_steps, self.num_visible+1)) # create random data container
		random_data[0,:-1] = np.random.rand(self.num_visible) # add random data to first row, the next rows will be populated in each iteration
		for step in range(num_steps):
			if step == num_steps-1:
				break
			# calculate hidden units states
			vis_states = random_data.astype(bool)
			hid_aes = np.dot(vis_states, self.weights)
			hid_probs = self._logistic_function(hid_aes)
			hid_probs[:,-1] = 1 # fix bias unit
			hid_states  = hid_probs > np.random.randn(random_data.shape[0], self.num_hidden+1)
			# calculate visible units states
			vis_aes = np.dot(hid_states, np.transpose(self.weights))
			vis_probs = self._logistic_function(vis_aes)
			vis_probs[:,-1] = 1 # fix bias unit
			vis_states  = vis_probs > np.random.randn(random_data.shape[0], self.num_visible+1)
			# set the newly calculated visible states as the next data row in the random_data
			random_data[step+1,:] = vis_states[step]
		return random_data[:,:-1]

	# Utility functions
	def _logistic_function(self, x):
		return 1 / (1 + np.exp(-x))

	def print_weights(self):
		print('')
		header = '/;'
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+';'
		header += 'BIAS'
		print(header)
		row_labels = ['INP_'+str(i) for i in range(self.num_visible)]+['BIAS']
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
