import numpy as np 
from scipy.io import savemat, loadmat

def compute_sim(emb1, emb2):
	emb1 = emb1.flatten()
	emb2 = emb2.flatten()
	from numpy.linalg import norm
	sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
		
	return sim

def compare(features_all, features_single):

	not_include = ['__version__', '__globals__', '__header__']

	correct = 0
	total = 0
	count = 0

	for key_all, value_all in features_all.items():

		if key_all not in not_include:

			print('==========================Processing ID==============================: ', count + 1)
			count = count + 1

			for emb1 in value_all:

				highest_sim = -100
				id_of_highest_sim = ''
				emb1 = np.expand_dims(emb1, axis=0)

				for key_single, value_single in features_single.items():

					if key_single not in not_include:

						emb2 = value_single
						sim = compute_sim(emb1, emb2)

						if sim > highest_sim:
							highest_sim = sim 
							id_of_highest_sim = key_single

				if key_all == id_of_highest_sim:
					correct = correct + 1

				total = total + 1

				print('Correct = ', correct)
				print('Total = ', total)

	return correct, total



if __name__ == '__main__':
	features_all = loadmat('./feature_map_files/feature_map_complete.mat')
	features_single = loadmat('./feature_map_files/feature_map_complete_single.mat')

	# print(features_all['Christian_Bale'][0])
	# print(features_single['Christian_Bale'].shape)

	correct, total = compare(features_all, features_single)

	print('Accuracy: ' , ((correct/total)*100))



