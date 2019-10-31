import cv2
import numpy as np
import os
import time

class processImage:
	def __init__(self, image):
		self.image = image

	def resize(self, new_size = (128, 128)):
		self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_CUBIC)
		return self.image


def get_hash_str(image_path):
	'''
	:param image_path: image_path
	:return: a tuple, the length of the tuple is 72
	'''
	# initialize Hue_list, Saturation_list and Value_list
	processImage.hsv_map()
	image = cv2.imread(image_path)
	obj = processImage(image)
	# resize the image into a standard size
	obj.resize()
	# generate the hash string of the image
	hash_str = obj.hsvHist()
	return hash_str


# if __name__ == "__main__":
# 	# testing
# 	dir = "../tmp/queryImage/query1"
# 	paths = os.listdir(dir)
# 	hash_list = []
# 	time_start = time.perf_counter()
# 	for i in paths:
# 		image_path = os.path.join(dir, i)
# 		hash_str = get_hash_str(image_path)
# 		hash_list.append(hash_str)
# 	time_end1 = time.perf_counter()
# 	print("It cost %d s !" % (time_end1 - time_start))

