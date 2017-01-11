# -*- coding: utf-8 -*-

import logging
import sys

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