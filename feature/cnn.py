'''
Here, we use convolution neural network(CNN) features to represent an image.
The CNN features are extracted by MobileNet.
The pre-trained MobileNet model is from tf.keras.applications API.
The following codes are using Tensorflow 2.0
'''
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

class CNN:

	def __init__(self):
		self.MobileNet = MobileNet
		self.preprocess_input = preprocess_input
		self.target_size = (224,224)

	def cnn_feature(self, image_array):
		self.model = self.MobileNet(input_shape = (224,224,3),
									include_top = False,
									pooling = 'avg')
		image_pp = self.preprocess_input(image_array)
		image_pp = np.array(image_pp)[np.newaxis, :]
		return self.model.predict(image_pp)



if  __name__ == "__main__":
	# testing code
	image_path = '../data/query.jpg'
	image = cv2.imread(image_path)
	image = cv2.resize(image, (224,224), interpolation = cv2.INTER_CUBIC)
	obj = CNN()
	feature_query = obj.cnn_feature(image)[0]

	import os
	os.chdir('../data/lib')
	best_score = 0.0
	best_image = ''
	all_score = []
	for img in os.listdir('.'):
		image = cv2.imread(img)
		image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
		obj = CNN()
		feature_base = obj.cnn_feature(image)[0]
		cur_score = np.dot(feature_query, feature_base)/(np.linalg.norm(feature_query) * np.linalg.norm(feature_base))
		if  cur_score > best_score:
			best_score = cur_score
			best_image = img
		all_score.append(cur_score)
	print(best_score)
	print(best_image)
	print(all_score)

