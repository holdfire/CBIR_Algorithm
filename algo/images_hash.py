import os
import cv2
import pickle
from image_preprocess import preprocessImage
from image_extractFeatures import extractImageFeatures

class hashImage:
	def __init__(self):
		# dictionary to store the hash value of all referenced images
		self.hash_table ={}

	def hash_batch_images(self, image_dir):
		# list all image path
		self.all_image_path = []
		for image_name in os.listdir(image_dir):
			image_absolute_path = os.path.join(image_dir, image_name)
			self.all_image_path.append(image_absolute_path)
		# hash each image
		for image_path in self.all_image_path:
			obj_prepro_img = preprocessImage(cv2.imread(image_path))
			obj_prepro_img.resize_image()
			obj_prepro_img.mask_image()
			obj_extract_img = extractImageFeatures(obj_prepro_img.image)
			extractImageFeatures.hsv_map()
			dHash_str = obj_extract_img.dHash()
			hsv_hist = obj_extract_img.hsvHist()
			self.hash_table[image_path] = (dHash_str, hsv_hist)

	def save_hash_table(self, path):
		hash_table_file = os.path.join(path, 'images_hash_table.pkl')
		with open(hash_table_file, "wb+") as f:
			pickle.dump(self.hash_table, f)
		return hash_table_file


