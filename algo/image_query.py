import cv2
import os
import time
import csv
import numpy as np
from image_preprocess import preprocessImage
from image_extractFeatures import extractImageFeatures

class queryImage:
	def __init__(self, image_path):
		self.image_path = image_path
		self.image = cv2.imread(image_path)

	def query_single_image(self, hash_table):
		# preprocess image
		obj_prepro_img = preprocessImage(self.image)
		obj_prepro_img.resize_image()
		obj_prepro_img.mask_image()
		# extract features of images
		obj_extract_img = extractImageFeatures(obj_prepro_img.image)
		extractImageFeatures.hsv_map()
		image_dHash = obj_extract_img.dHash()
		image_hsvHist = obj_extract_img.hsvHist()

		best_score = np.zeros((4))
		best_ref_image_path = ''
		for item in hash_table.items():
			ref_image_path = item[0]
			# calculate the score of the image_path and the ref_image_path
			score_dHash = self.cmpHash_str(image_dHash, item[1][0])
			score_hsvHist = self.cmpHist(image_hsvHist, item[1][1])
			if score_dHash > best_score[0]:
				best_score = [score_dHash, score_hsvHist]
				best_ref_image_path = ref_image_path
		# save the query log
		queryImage.query_log.append([self.image_path, best_ref_image_path, best_score[0], best_score[1]])
		return best_ref_image_path, best_score[0]

	def cmpHash_str(self, hash_str1, hash_str2):
		n = 0
		if len(hash_str1) != len(hash_str2):
			raise Exception("The input hash strings do not match")
		for i in range(len(hash_str1)):
			if hash_str1[i] != hash_str2[i]:
				n = n + 1
		score = 1 - n / len(hash_str1)
		return score

	def cmpHist(self, hist1, hist2):
		return cv2.compareHist(hist1,hist2,cv2.HISTCMP_INTERSECT) / np.sum(hist1)

	# a class variable to store the query log
	query_log = []
	score_threshold = 0.7
	# a class method to save the query log
	@classmethod
	def cls_save_query_result_log(cls, dirname):
		query_log_dir = os.path.join(dirname, 'query_result_log')
		if not os.path.exists(query_log_dir):
			os.mkdir(query_log_dir)
		query_log_file = os.path.join(query_log_dir, 'query_result_log_' + time.strftime('%Y%m%d%H%M%S') + '.csv')
		with open(query_log_file, 'w+', newline='') as f:
			f_csv = csv.writer(f)
			f_csv.writerows(cls.query_log)

	@classmethod
	def cls_classify_queried_image(cls, dirname):
		regular_dir = os.path.join(dirname, "retrieved_regular")
		if not os.path.exists(regular_dir):
			os.mkdir(regular_dir)
		irregular_dir = os.path.join(dirname, "retrieved_irregular")
		if not os.path.exists(irregular_dir):
			os.mkdir(irregular_dir)
		# for each queried image, judge whether it exists in the referenced image repository
		failed_image_path = []
		for item in cls.query_log:
			image_path = item[0]
			basename = os.path.basename(image_path)
			score_dHash = item[2]
			score_hsvHist = item[3]
			image = cv2.imread(image_path)
			if ((score_dHash >= queryImage.score_threshold)  & (score_hsvHist >= queryImage.score_threshold)):
				cv2.imwrite(os.path.join(regular_dir, basename), image)
			else:
				failed_image_path.append(image_path)
				cv2.imwrite(os.path.join(irregular_dir, basename), image)
		return failed_image_path







