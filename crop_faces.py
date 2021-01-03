import os
import sys
import numpy as np 
import cv2
import insightface
from scipy.io import savemat, loadmat

def load_detection_model():
	print('Loading and preparing detection Model')
	detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
	detector.prepare(ctx_id = -1)

	return detector

def crop_images(input_dir, output_dir):

	detector = load_detection_model()
	
	for count, identity in enumerate(os.listdir(input_dir)):

		print('Processing identity: ', count + 1)

		if count == 100:
			break

		identity_input_dir = os.path.join(input_dir, identity)
		identity_output_dir = os.path.join(output_path, identity)

		if not os.path.exists(identity_output_dir):
			os.mkdir(identity_output_dir)

		for sub_dir in os.listdir(identity_input_dir):
			sub_dir_input = os.path.join(identity_input_dir, sub_dir)
			sub_dir_output = os.path.join(identity_output_dir, sub_dir)

			if not os.path.exists(sub_dir_output):
				os.mkdir(sub_dir_output)

			for filename in os.listdir(sub_dir_input):
				img_input_path = os.path.join(sub_dir_input, filename)
				img_output_path = os.path.join(sub_dir_output, filename)

				
				try:
					img = cv2.imread(img_input_path)
					img = cv2.resize(img, (400,400))


					bbox, landmark = detector.detect(img, threshold=0.5, scale=1.0)
					face = bbox[0]

					startX = int(face[0])
					startY = int(face[1])
					endX = int(face[2])
					endY = int(face[3])
					img = img[startY:endY, startX:endX]

					img = cv2.resize(img, (112,112))


					cv2.imwrite(img_output_path, img)

				except:
					continue





if __name__ == '__main__':
	dataset_path = '/home/sohaibjawad/Desktop/datasets/aligned_images_DB/'
	output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cropped_images')

	crop_images(dataset_path, output_path)


