import os
import pickle as pk
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


class Model(object):

	def __init__(self):
		self._params = None
		self._data = None
		self._trained_model = None
		self._base_model = None


	def save_model(self,save_path=None,model_name=None,prefix=None):
		if not self._trained_model:
			raise Exception("NO MODEL FOUND!!")
		if not model_name:
			model_name = self._create_model_name()
		if prefix:
			model_name = prefix + model_name
		complete_path = save_path + model_name
		with open(complete_path,"wb") as out:
			pk.dump(self._trained_model,out)
		print("Model {} save in {}".format(model_name,save_path))


	def load_model(self,save_path=None,model_name=None):
		with open(save_path + model_name,"rb") as inp:
			self._trained_model = pk.load(inp)
		print("Model {} loaded from {}".format(model_name,save_path))


	def _create_model_name(self):
		params = self._trained_model.get_params()
		name = ""
		name_thresh = 5
		for k,v in params.items():
			name += k[:name_thresh] + ":" + str(v) + "%"
		return name


	def get_model(self):
		return self._trained_model


	def set_model(self,model):
		self._trained_model = model


	def set_data(self,data):
		self._data = data


	def get_data(self):
		return self._data


	def set_base_model(self,m):
		self._base_model = m


	def get_base_model(self):
		return self._base_model


	def get_params(self):
		return self._params


class sample_model(Model):

	def __init__(self):
		super().__init__()
		self._trained_model = 2
	def get_model(self):
		return self._trained_model



def main():
	s = sample_model()
	print(s.get_model())
	print(dir(s))


if __name__ == '__main__':
	main()