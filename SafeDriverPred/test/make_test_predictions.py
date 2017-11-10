import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pk
import datetime

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
UTIL_DIR = CURRENT_DIR + "../util/"
sys.path.append(UTIL_DIR)
from data_provider import DataProvider

MODEL_DIR = CURRENT_DIR + "trainedModels/"

IGNORE_FEATURES = ["id","target"]



def get_model(path):
	with open(path,"rb") as inp:
		return pk.load(inp)


def write_pred_to_file(preds):
	pass


def main():
	dp = DataProvider()
	test_data = dp.get_test_data()
	model_name="rando:0%reg_a:0%max_d:0%subsa:1%boost:gbtree%nthre:8%colsa:1%learn:0.025%scale:5.2872645858027125%max_d:3%missi:None%gamma:0%base_:0.5%colsa:1%min_c:2%seed:100%n_job:1%silen:0%n_est:800%reg_l:1%objec:binary:logistic%"
	path="/home/msaffarm/KaggleChallenges/SafeDriverPred/xgbModel/trainedModels/" + model_name
	model = get_model(path)
	test_ids = test_data[["id"]].as_matrix()
	test_data.drop(["id"], axis=1,inplace=True)
	preds = model.get_booster().predict(xgb.DMatrix(test_data))
	final_pred = np.concatenate([test_ids.reshape(-1,1),preds.reshape(-1,1)],axis=1)
	final_pred_df = pd.DataFrame(final_pred,columns=["id","target"])
	final_pred_df["id"] = final_pred_df["id"].astype(int)
	print(final_pred_df)

	final_pred_df.to_csv("predictions.csv",index=False)



if __name__ == '__main__':
	main()