import sys
import numpy as np 
import cv2
import insightface
from scipy.io import savemat, loadmat
import os

def load_recognition_model():
	print('Loading and preparing Recognition Model')
	recognizer = insightface.model_zoo.get_model('model_r100_ii')
	recognizer.prepare(ctx_id = -1)

	return recognizer

def extract_features(input_dir):

	global_feature_map = dict()

	recognizer = load_recognition_model()

	for count, identity in enumerate(os.listdir(input_dir)):

		identity_features_list = []

		print('Processing ID: ', count + 1)

		identity_input_dir = os.path.join(input_dir, identity)

		for sub_dir in os.listdir(identity_input_dir):
			sub_input_dir = os.path.join(identity_input_dir, sub_dir)

			for filename in os.listdir(sub_input_dir):
				img_input_path = os.path.join(sub_input_dir, filename)

				img = cv2.imread(img_input_path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				embedding = recognizer.get_embedding(img)
				identity_features_list.append(embedding[0])

		global_feature_map[identity] = identity_features_list

	return global_feature_map








if __name__ == '__main__':
	cropped_dataset_path = './cropped_images/'
	output_path = './feature_map_files/'

	feature_map = extract_features(cropped_dataset_path)

	savemat(output_path + 'feature_map_100.mat', feature_map)