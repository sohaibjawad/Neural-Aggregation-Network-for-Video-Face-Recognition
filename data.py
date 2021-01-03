from scipy.io import savemat, loadmat
import random

class Data():

	def __init__(self, filename, batch_size, class_num):

		self.filename = filename
		self.batch_size = batch_size
		self.class_num = class_num


	def load_feature(self):

		self.features = []
		self.labels = []
		
		dataset = loadmat(self.filename)
		not_include = ['__version__', '__globals__', '__header__']
		flag = 0

		for key, value in dataset.items():
			if key not in not_include:
				label = [0] * self.class_num
				label[flag] = 1
				flag = flag + 1

				sub_features = []

				for idx in range(len(value)):
					sub_features.append(value[idx])
				self.labels.append(label)
				self.features.append(sub_features)


	def next_batch(self, group_num):

		train_feature, train_label = [], []
		start = random.randint(0, self.class_num)

		for i in range(start, start + self.batch_size):
			train_group = []
			seed = random.randint(0, len(self.features[i % self.class_num]) - group_num)

			for j in range(seed, seed + group_num):
				train_group.append(self.features[i % self.class_num][j])

			train_feature.append(train_group)
			train_label.append(self.labels[i % self.class_num])

			#train_feature = train_feature.reshape(self.batch_size, self.group_num, 512)

		return train_feature, train_label



if __name__ == '__main__':

	filename = './feature_map_files/feature_map_100.mat'
	dataset = Data(filename, 3, 100)
	dataset.load_feature()
	train_features, train_label = dataset.next_batch(5)

	print(len(train_features))

	print(len(train_label[0]))






