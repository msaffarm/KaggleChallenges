import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
DATA_DIR = CURRENT_DIR + "../data/"

TEST_TRAIN_RATIO = 0.3
RANDOM_STATE = 100

class DataProvider(object):

	def __init__(self):

		self._data = None
		self._train_data = None
		self._train_label = None
		self._test_data = None
		self._test_label = None
		self._sampled_datasets = None


	def clean_data(self,data):
		data.replace(-1,np.nan,inplace=True)


	def get_all_features(self):
		return(list(self._data.columns.values))


	def read_data(self,file):
		self._data =  pd.read_csv(DATA_DIR + file)
		self.clean_data(self._data)


	def get_test_data(self):
		data = Wpd.read_csv(DATA_DIR + "test.csv")
		self.clean_data(data)
		features = list(data.columns.values)
		self._data = self.vectorize_data(data,features=features)
		return self._data


	def get_data(self):
		return self._data


	def vectorize_data(self,data,features=None,target="target"):

		categorical_features = [x for x in features if "cat" in x or "bin" in x]
		if target in features:
			features.remove(target)
		X_vectorized = pd.get_dummies(data[features],
			columns=categorical_features,prefix_sep='_')
		return X_vectorized



	def _create_test_train_data(self,data,features=None,target="target"):

		X_vectorized = self.vectorize_data(data,features=features,target=target)
		y = data[[target]]
	
		self._train_data,self._test_data,self._train_label,self._test_label=\
		train_test_split(X_vectorized,y,test_size=TEST_TRAIN_RATIO,\
			random_state=RANDOM_STATE,stratify=y)

		self._train_label = np.ravel(self._train_label)
		self._test_label = np.ravel(self._test_label)



	def _create_test_train_data_resample(self,features=None,target="target",
		ratio=1):

		X_vectorized = self.vectorize_data(self._data,features=features,target=target)
		y = self._data[[target]]

		pos_index = np.array(self._data.index[self._data["target"]==1])
		neg_index = np.array(self._data.index[self._data["target"]==0])

		num_of_partitions = len(neg_index)//(len(pos_index)*ratio)
		
		np.random.seed(RANDOM_STATE)
		np.random.shuffle(neg_index)
		neg_splits = np.array_split(neg_index,num_of_partitions)

		sampled_datasets = []
		for split in neg_splits:
			indices = np.concatenate([split,pos_index])
			y_split = y.loc[indices].copy()
			X_split = X_vectorized.loc[indices].copy()
			train_data,test_data,train_label,test_label= train_test_split(
				X_split,y_split,test_size=TEST_TRAIN_RATIO,
				random_state=RANDOM_STATE,stratify=y_split)
			train_label = np.ravel(train_label)
			test_label = np.ravel(test_label)
			sampled_datasets.append((train_data,test_data,train_label,test_label))

		self._sampled_datasets = sampled_datasets



	def get_test_train_data(self,features=None,target="target",method="all_data"
		,ratio=1,force_create=False):

		if method=="all_data":
			if not self._train_data or force_create:
				self._create_test_train_data(self._data,features=features,target=target)
			return self._train_data,self._test_data,self._train_label,self._test_label
		
		elif method=="resample":
			if not self._sampled_datasets or force_create:
				self._create_test_train_data_resample(features=features,target=target,
					ratio=ratio)
			return self._sampled_datasets


def main():
	dp = DataProvider()
	dp.read_data("train.csv")
	data = dp.get_data()
	cols = list(data.columns.values)
	cols.remove("id")
	datasets = dp.get_test_train_data(features=cols,
		method="resample")
	print(len(datasets))



if __name__ == '__main__':
	main()