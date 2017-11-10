import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pk
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import datetime
from hyperopt import hp,STATUS_OK

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider
from measurement import Measurement
from model import Model
from optimizer import Optimizer

MODEL_DIR = CURRENT_DIR + "trainedModels/"

IGNORE_FEATURES = ["id","target"]


class XGBClassifier(Model):

	def __init__(self):
		super().__init__()


	def get_train_test_Dmat(self):
		X_train,X_test,y_train,y_test = self._data
		train_dmat = xgb.DMatrix(X_train,label=y_train)
		test_dmat = xgb.DMatrix(X_test,label=y_test)
		return train_dmat, test_dmat



	def tune_hyperparams_scikit(self,model,X_train,y_train):
		"""
		Tuning parameters of model using scikit learn GridSearch
		"""
		print("Started CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		
		# create gini scorer
		measurer = Measurement()
		gini_score = measurer.get_gini_scorer()

		params = [{"learning_rate":[0.01],
		"n_estimators":[100,200],
		"seed":[100],
		"max_depth":[2],
		"min_child_weight":[1,5],
		"subsample":[1]
		}]

		print("Running GridSearch")
		gscv = GridSearchCV(model,params,cv=2,n_jobs=-1,
			scoring=gini_score,verbose=3,error_score=0)
		gscv.fit(X_train,y_train)
		best_model = gscv.best_estimator_
		# save best model
		save_model(best_model,MODEL_DIR,prefix="CV5-bestModel-")
		# save CV results
		gscv_resutls = pd.DataFrame(gscv.cv_results_)
		gscv_resutls.to_csv("GridSearchRestest.csv",index=False)
		
		print("Finished CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))



	def xgbcv(self,X,y,params=None):
		"""
		Tuning parameters of model using XGBoost CVs
		"""
		print("Started CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		
		# create gini scorer
		measurer = Measurement()
		gini_score = measurer.xgb_gini_scorer
		# create StratifiedKFold object
		skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=100)
		# dmat
		dtrain = xgb.DMatrix(X,label=y)
		if not params:
			params = {"learning_rate":0.1,"n_estimators":20,"seed":100,
			"max_depth":3,"min_child_weight":1,"subsample":1,
			"tree_method":"gpu_hist","predictor":"cpu_predictor",
			"objective":'binary:logistic'}

		cv_results = xgb.cv(params,dtrain,num_boost_round=int(params["n_estimators"]),
			feval=gini_score,folds=list(skf.split(X,y)),verbose_eval=5)
		cv_results.to_csv("cv_resutls.csv",index=False)
		
		print("Finished CV task at "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


	def make_predictions(self,trained_model,data):
		X_train,X_test,y_train,y_test = data
		#get predictions
		print(type(trained_model))
		print("Making predictions!")
		train_pred = trained_model.predict(X_train)
		train_pred_prob = trained_model.predict_proba(X_train)
		test_pred = trained_model.predict(X_test)
		test_pred_prob = trained_model.predict_proba(X_test)
		print(test_pred_prob)
		# print metrics
		measurer = Measurement()
		print("Train resutls:")
		measurer.print_measurements(y_train,train_pred,train_pred_prob[:,1])
		print("Test resutls:")
		measurer.print_measurements(y_test,test_pred,test_pred_prob[:,1])		



def create_xgbmodel(dp,xgb_model,device="cpu"):
	X_train,X_test,y_train,y_test = get_data(dp)
	xgb_model.set_data((X_train,X_test,y_train,y_test))

	saved_model = "CV5-bestModel-nest=200-lr=0.1-m3"
	if not os.path.exists(MODEL_DIR + saved_model):
		print("Training new model")
		n2p_ratio = len(y_train[y_train==0])/len(y_train[y_train==1])
		# check for training device
		if device=="gpu":
			tree_method = "gpu_hist"
			predictor="cpu_predictor"
		else:
			tree_method = "auto"
			predictor="cpu_predictor"
		# create model
		xgb_model_base = xgb.XGBClassifier(learning_rate=0.01,min_child_weight=2,
			n_estimators=1000,silent=0,seed=100,max_depth=4, 
			scale_pos_weight=n2p_ratio/5,nthread=8,objective='binary:logistic',
			predictor=predictor,tree_method=tree_method)
		xgb_model.set_base_model(xgb_model_base)

		# cv task
		print("Performing Cross Validation task")
		X = np.concatenate([X_train,X_test])
		y = np.concatenate([y_train,y_test])
		xgb_model.xgbcv(X,y)
		return

		# fit model
		print("Fitting model")
		trained_model = xgb_model.get_base_model().fit(X_train,y_train)
		xgb_model.set_model(trained_model)
		# save model
		xgb_model.save_model(save_path=MODEL_DIR)

	# else load model
	else:
		train_model = xgb_model.load_model(save_path=MODEL_DIR,model_name=saved_model)
		xgb_model.set_model(trained_model)

	# make predictions
	xgb_model.make_predictions(xgb_model.get_model(),xgb_model.get_data())


def get_data(dp):
	# extract features to be used in classifier
	features = dp.get_all_features()
	for x in IGNORE_FEATURES:
		features.remove(x)
	print("Reading test/train data")
	# X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features)
	X_train,X_test,y_train,y_test = dp.get_test_train_data(features=features,
		method="all_data")
	return (X_train,X_test,y_train,y_test)


def objective(params):
	xgb_model,measurer = params["xgb_model"],params["measurer"]
	train_dmat,test_dmat = xgb_model.get_train_test_Dmat()
	trained_model = xgb.train(params,train_dmat,
		num_boost_round=int(params["n_estimators"]))
	test_pred = trained_model.predict(test_dmat)
	test_label = xgb_model.get_data()[-1]
	loss = -measurer.normalized_gini(test_label,test_pred)
	print("Loss= {} and Params= {} ".format(loss,params))
	return {"loss":loss,"status":STATUS_OK}


def objective_with_cv(params):
	xgb_model,measurer = params["xgb_model"],params["measurer"]
	dtrain = xgb_model.get_data()
	cv_res = xgb.cv(params,dtrain,num_boost_round=int(params["n_estimators"]),
		feval=measurer.xgb_gini_scorer,folds=params["folds"])
	loss = cv_res.iloc[-1]["test-normalized_gini-mean"]	
	print("Loss= {} and Params= {} ".format(loss,params))
	return {"loss":loss,"status":STATUS_OK}	


def tune_with_TPE(dp,xgb_model,opt):
	"""
	Hyperparameter tuning using TPE:
	https://github.com/hyperopt/hyperopt/wiki/FMin

	for XGBOOST params doc see:
	https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
	"""
	# add xgb_model,measurer
	X_train,X_test,y_train,y_test = get_data(dp)
	xgb_model.set_data((X_train,X_test,y_train,y_test))
	n2p_ratio = len(y_train[y_train==0])/len(y_train[y_train==1])

	# set optimizer parameters
	opt.set_max_eval(2)

	# # add params to handle cv task
	# skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=100)
	# X = np.concatenate([X_train,X_test])
	# y = np.concatenate([y_train,y_test])
	# xgb_model.set_data(xgb.DMatrix(X,label=y))

	# define search space
	search_space = {"n_estimators": hp.quniform("n_estimators",100,1500,50),
	"eta": hp.quniform("eta",0.025,0.1,0.025),
	"max_depth": hp.choice("max_depth",[3,4]),
	"min_child_weight": hp.quniform('min_child_weight',1,4,1),
	"subsample": hp.quniform("subsample",0.9,1.0,0.1),
	"colsample_bytree": hp.quniform("colsample_bytree",0.8,1.0,0.1),
	"gamma": hp.quniform("gamma",0,0.2,0.05),
	"scale_pos_weight": hp.quniform("scale_pos_weight",\
		n2p_ratio/10,n2p_ratio,n2p_ratio/10),
	# "alpha": hp.choice("alpha",[1e-5,1e-2,1,10]),
	"nthread":8,
	"silent":1,
	"seed":100,
	"measurer": Measurement(),
	"xgb_model": xgb_model,
	# "folds":list(skf.split(X,y)),
	# "tree_method":"auto",
	# "predictor":"cpu_predictor"
	}



	opt.set_search_space(search_space)
	opt.set_objective(objective)
	# start optimization
	best = opt.optimize()
	opt.save_trials(CURRENT_DIR)
	
	# retrain and save best model
	train_dmat,_ = xgb_model.get_train_test_Dmat()
	best_model = xgb.train(best,train_dmat,
		num_boost_round=int(best["n_estimators"]))
	xgb_model.set_model(best_model)
	xgb_model.save_model(save_path=MODEL_DIR,model_name=str(best),
		prefix="TPE_200_BestModel")


def main():
	# read data
	dp = DataProvider()
	xgb_model = XGBClassifier()
	dp.read_data("train.csv")
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	# create_xgbmodel(dp,xgb_model,device="gpu")
	opt = Optimizer()
	tune_with_TPE(dp,xgb_model,opt)


if __name__ == '__main__':
	main()
