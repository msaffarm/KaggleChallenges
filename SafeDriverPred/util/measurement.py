import os
import sys
import numpy as np
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,\
make_scorer

class Measurement(object):

	def print_measurements(self,true_label,pred_label,pred_prob):
		print("ACCURACY= {}".format(self.accu(true_label,pred_label)))
		print("AUC= {}".format(self.AUC(true_label,pred_prob)))
		print("NORMALIZED GINI= {}".format(self.normalized_gini(true_label,pred_prob)))
		print("SKLEARN REPORT:")
		self.sklearn_report(true_label,pred_label)


	def AUC(self,true_label,pred_prob):
		return roc_auc_score(true_label, pred_prob)


	def accu(self,true_label,pred_label):
		return accuracy_score(true_label,pred_label)


	def sklearn_report(self,true_label,pred_label,
		classes=["negative(0)","positive(1)"]):
		print(classification_report(true_label,pred_label,target_names=classes))

	def gini(self,true,pred):
		concat = np.c_[true,pred,range(len(true))]
		idx = np.lexsort((concat[:,0],-concat[:,1]))
		sorted_concat = concat[idx]
		gini_sum = sorted_concat[:,0].cumsum().sum()/sorted_concat[:,0].sum()
		gini_sum -= float((len(true)+1)/2)
		return gini_sum


	def normalized_gini(self,true,pred):
		return self.gini(true,pred)/self.gini(true,true)

	def _normalized_gini_sklearn(self,true,pred):
		return self.gini(true,pred[:,1])/self.gini(true,true)

	def get_gini_scorer(self):
		return make_scorer(self._normalized_gini_sklearn,needs_proba=True,
			greater_is_better=True)

	def xgb_gini_scorer(self,pred,dtrain):
		labels = dtrain.get_label()
		# print(labels)
		# print(pred)
		return "normalized_gini" , self.normalized_gini(labels,pred)


