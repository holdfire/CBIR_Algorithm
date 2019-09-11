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
		image_dHash = obj_extract_img.dHash()
		image_pHash = obj_extract_img.pHash()
		image_grayHist = obj_extract_img.grayHist()
		image_hsvHist = obj_extract_img.hsvHist()
		image_bgrHist = obj_extract_img.bgrHist()

		best_score = np.zeros((4))
		best_ref_image_path = ''
		for item in hash_table.items():
			ref_image_path = item[0]
			# calculate the score of the image_path and the ref_image_path
			score_dHash = self.cmpHash_str(image_dHash, item[1][0])
			score_pHash = self.cmpHash_str(image_pHash, item[1][1])
			score_grayHist = self.cmpHist(image_grayHist, item[1][2])
			score_hsvHist = self.cmpHist(image_hsvHist, item[1][3])
			score_bgrHist = self.cmpHist(image_bgrHist, item[1][4])
			if score_dHash > best_score[0]:
				best_score = [score_dHash, score_pHash, score_grayHist, score_hsvHist, score_bgrHist]
				best_ref_image_path = ref_image_path
		# save the query log
		queryImage.query_log.append([self.image_path, best_ref_image_path, best_score[0], best_score[1], best_score[2], best_score[3], best_score[4]])
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
		if not len(hist1) == len(hist2):
			raise Exception("The input hist length should be the same")
		inter = 0
		total = 0.01
		for i in range(len(hist1)):
			inter = inter + min(hist1[i], hist2[i])
			total = total + max(hist1[i], hist2[i])
		score = (inter/total)
		return score

	def cmpHist2(self, hist1, hist2):
		if not len(hist1) == len(hist2):
			raise Exception("The input hist length should be the same")
		degree = 0
		for i in range(len(hist1)):
			if (hist1[i] != hist2[i]):
				degree = degree + 1 - (abs(hist1[i] - hist2[i]) / max(hist1[i] , hist2[i]))
			else:
				degree = degree + 1
		score = degree / len(hist1)
		return score

	# a class variable to store the query log
	query_log = []
	score_dHash_threshold = 0.6
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
			score_pHash = item[3]
			score_grayHist = item[4]
			score_hsvHist = item[5]
			score_bgrHist = item[6]
			image = cv2.imread(image_path)
			if ((score_dHash >= 0.6) & (score_pHash >= 0.7) & (score_grayHist >= 0.6) & (score_hsvHist >= 0.6) & (score_bgrHist >= 0.6)):
				cv2.imwrite(os.path.join(regular_dir, basename), image)
			else:
				failed_image_path.append(image_path)
				cv2.imwrite(os.path.join(irregular_dir, basename), image)
		return failed_image_path







