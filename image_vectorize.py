from features.hist import Hist
import cv2
import numpy as np
import os
import time

class Vectorize:
	def __init__(self, image):
		self.image = image

	def resize(self, new_size=(128, 128)):
		self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_CUBIC)

	def get_vector(self):
		'''
		:param image_path: image_path
		:return: a tuple, the length of the tuple is 72
		'''
		self.resize()
		# initialize Hue_list, Saturation_list and Value_list
		Hist.hsv_map()
		obj = Hist(self.image)
		vec = obj.hsv_hist()
		return vec


if __name__ == "__main__":
	image_path = "./data/query.jpg"
	image = cv2.imread(image_path)
	obj = Vectorize(image)
	vec = obj.get_vector()
	print(vec)
