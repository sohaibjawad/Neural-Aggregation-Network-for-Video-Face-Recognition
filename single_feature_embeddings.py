from scipy.io import savemat, loadmat
import numpy as np 
import tensorflow as tf 


def single_features(features_all, model):

	single_features = dict()
	count = 1

	not_include = ['__version__', '__globals__', '__header__']

	for key, value in features_all.items():
		if key not in not_include:

			print('Processing ID: ', count)
			count = count + 1

			input_x = np.array(features_all[key]) # (num of features of a person, 512)

			with tf.Session() as sess:    
				model.restore(sess,tf.train.latest_checkpoint('./model/'))
				q0 = sess.run('q0:0')


			q0 = np.expand_dims(q0, 1) # (512, 1)
			ek = np.dot(input_x, q0) # (num of features of a person,1)
			ak = np.exp(ek)/sum(np.exp(ek))

			r0 = np.dot(ak.T, input_x) # (1, num of features of a person) * (num of features of a person, 512) = (1, 512) 

			single_features[key] = r0

	return single_features

			



if __name__ == '__main__':
	features_all = loadmat('./feature_map_files/feature_map_complete.mat')
	model = tf.train.import_meta_graph('./model/attention.ckpt-80000.meta')

	features_single = single_features(features_all, model)

	savemat('./feature_map_files/feature_map_complete_single.mat', features_single)
