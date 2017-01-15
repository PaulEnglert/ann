# -*- coding: utf-8 -*-

import logging
import sys
import numpy as np

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

	@staticmethod
	def create_minibatches(data, size):
		batches=[]
		batch = []
		for i in range(len(data)):
			batch.append(data[i])
			if len(batch) == size or i == len(data)-1:
				batches.append(np.asarray(batch))
				batch = []
		if len(batches[-1]) < size:
			return batches[0:-2]
		return batches