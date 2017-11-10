import os
import sys
import pickle as pk
from hyperopt import hp,fmin,tpe, STATUS_OK, Trials
import datetime


class Optimizer(object):
	"""
	Hyperparameter Optimization using hyperopt bayesian optimization
	package. 
	"""

	def __init__(self):
		self._space = None
		self._trials = Trials()
		self._objective = None
		self._maxeval = 10


	def optimize(self):
		"""
		Run optimization using the objective function 
		"""
		print("Starting optimization "+
			datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		best = fmin(fn=self._objective,
			algo=tpe.suggest,
			space=self._space,
			max_evals=self._maxeval,
			trials=self._trials
			)
		print("Done Optimizing at {}, best values is {}".format(
			datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),best))
		return best


	def save_trials(self,save_path):
		with open(save_path + "trials_dump.pkl",'wb') as out:
			pk.dump(self._trials,out)
		print("trails dumped successfully in {}".format(save_path))


	def get_trials(self,trial_path=None):
		if self._trials:
			return self._trials
		if trial_path:
			with open(save_path + "trials_dump.pkl",'rb') as inp:
				return pk.load(inp)


	def set_search_space(self,space):
		self._space = space


	def set_max_eval(self,v):
		self._maxeval = v


	def set_objective(self,obj):
		self._objective = obj