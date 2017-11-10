import os
import sys
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import datetime
from hyperopt import hp,STATUS_OK
import lightgbm as lgb


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider
from measurement import Measurement
from model import Model
from optimizer import Optimizer

MODEL_DIR = CURRENT_DIR + "trainedModels/"

IGNORE_FEATURES = ["id","target"]



def create_model(dp):
	# create train/validation data
	df = dp.get_data()
	y = df[["target"]].as_matrix()
	df.drop(["id","target"],inplace=True,axis=1)
	features = list(df.columns.values)
	categorical_features = [x for x in features if "cat" in x or "bin" in x]





def main():
	# read data
	dp = DataProvider()
	dp.read_data("train.csv")


	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	create_model(dp)



if __name__ == '__main__':
	main()