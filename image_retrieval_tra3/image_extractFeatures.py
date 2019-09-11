import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_preprocess import preprocessImage

class extractImageFeatures:
	def __init__(self, image):
		# - parameter: an image read by cv2.imread()
		self.image = image

	def my_bgr2gray(self):
		channels = cv2.split(self.image)
		bgr_coef = [0.3, 0.59, 0.11]
		gray_image = channels[0] * bgr_coef[0] + channels[1] * bgr_coef[1] + channels[2] * bgr_coef[2]
		return gray_image

	def dHash(self, hash_size=(17, 16)):
		gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		resized_gray_image = cv2.resize(gray_image, hash_size, interpolation=cv2.INTER_CUBIC)
		dHash_str = ''
		for i in range(hash_size[1]):
			for j in range(hash_size[1]):
				if resized_gray_image[i][j] >= resized_gray_image[i][j + 1]:
					dHash_str = dHash_str + "1"
				else:
					dHash_str = dHash_str + "0"
		return dHash_str

	def pHash(self):
		gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		hash_size = (32, 32)
		resized_gray_image = cv2.resize(gray_image, hash_size, interpolation=cv2.INTER_CUBIC)
		# build a 2D array
		h, w = resized_gray_image.shape[:2]
		vis0 = np.zeros((h, w), np.float32)
		vis0[:h, :w] = resized_gray_image
		# 2D DCT
		image_dct = cv2.dct(cv2.dct(vis0))
		# compute the mean value
		avg = np.mean(image_dct[:8][:8])
		arr = image_dct[:8][:8] - avg
		pHash_str = ''
		for i in range(8):
			for j in range(8):
				if arr[i][j] > 0:
					pHash_str = pHash_str + '1'
				else:
					pHash_str = pHash_str + '0'
		return pHash_str

	def grayHist(self, hist_nums=[16], hist_range=[0, 256]):
		gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		hist_gray = cv2.calcHist([gray_image], [0], None, hist_nums, hist_range).reshape((1,-1))[0]
		hist_gray = extractImageFeatures.smoothHist(hist_gray)
		return hist_gray

	def hsvHist(self, hist_nums=([30],[16],[16]), hist_range=([0,180], [0,256], [0, 256])):
		image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
		b, g, r = cv2.split(image)
		hist_h = cv2.calcHist([b], [0], None, hist_nums[0], hist_range[0]).reshape((1, -1))[0]
		hist_s = cv2.calcHist([g], [0], None, hist_nums[1], hist_range[1]).reshape((1, -1))[0]
		hist_v = cv2.calcHist([r], [0], None, hist_nums[2], hist_range[1]).reshape((1, -1))[0]
		# for the hsv space, the hue value is in a loop between [0,180]
		hist_h = extractImageFeatures.smoothHist(hist_h, True)
		hist_s = extractImageFeatures.smoothHist(hist_s)
		hist_v = extractImageFeatures.smoothHist(hist_v)
		hist_hsv = np.hstack([hist_h, hist_s, hist_v])
		return hist_hsv

	def bgrHist(self, hist_nums=[16], hist_range=[0, 256]):
		b, g, r = cv2.split(self.image)
		hist_b = cv2.calcHist([b], [0], None, hist_nums, hist_range).reshape((1, -1))[0]
		hist_g = cv2.calcHist([g], [0], None, hist_nums, hist_range).reshape((1, -1))[0]
		hist_r = cv2.calcHist([r], [0], None, hist_nums, hist_range).reshape((1, -1))[0]
		hist_b = extractImageFeatures.smoothHist(hist_b)
		hist_g = extractImageFeatures.smoothHist(hist_g)
		hist_r = extractImageFeatures.smoothHist(hist_r)
		hist_bgr = np.hstack([hist_b, hist_g, hist_r])
		return hist_bgr

	@staticmethod
	def smoothHist(hist, loop = False):
		new_hist = np.zeros((len(hist)))
		for i in range(len(hist)):
			if i == 0:
				if loop:
					new_hist[i] = (hist[len(hist)-1] + hist[0] + hist[1]) / 3
				else:
					new_hist[i] = (hist[0] + hist[1]) / 2
			if i == len(hist)-1:
				if loop:
					new_hist[i] = (hist[i-1] + hist[i] + hist[0]) / 3
				else:
					new_hist[i] = (hist[i-1] + hist[i]) / 2
			else:
				new_hist[i] = (hist[i-1] + hist[i] + hist[i+1])/3
		return new_hist


if __name__ == "__main__":

	# testing code
	def cmpHash_str(hash_str1, hash_str2):
		n = 0
		if len(hash_str1) != len(hash_str2):
			raise Exception("The input hash strings do not match")
		for i in range(len(hash_str1)):
			if hash_str1[i] != hash_str2[i]:
				n = n + 1
		score = 1 - n / len(hash_str1)
		return score

	def cmpHist(hist1, hist2):
		if not len(hist1) == len(hist2):
			raise Exception("The input hist length should be the same")
		inter = 0.0
		total = 0.01
		for i in range(len(hist1)):
			inter = inter + min(hist1[i], hist2[i])
			total = total + max(hist1[i], hist2[i])
		score = inter / total
		return score


	path1 = "C:\\Users\\Dell\\Desktop\\omi_ori_data\\1.jpg"
	path2 = "C:\\Users\\Dell\\Desktop\\omi_ori_data\\2.jpg"
	image1 = cv2.imread(path1)
	obj1_prepro = preprocessImage(image1)
	obj1_prepro.resize_image()
	obj1_prepro.mask_image()
	obj1_extract = extractImageFeatures(obj1_prepro.image)
	hist1 = obj1_extract.grayHist()
	hist11 = obj1_extract.hsvHist()
	hist111 = obj1_extract.bgrHist()

	image2 = cv2.imread(path2)
	obj2_prepro = preprocessImage(image2)
	obj2_prepro.resize_image()
	obj2_prepro.mask_image()
	obj2_extract = extractImageFeatures(obj2_prepro.image)
	hist2 = obj2_extract.grayHist()
	hist22 = obj2_extract.hsvHist()
	hist222 = obj2_extract.bgrHist()

	print(cmpHist(hist1, hist2))
	print(cmpHist(hist11, hist22))
	print(cmpHist(hist111, hist222))
	# print(cmpHash_str(str1, str2))
	cv2.imshow('1.jpg', obj1_prepro.image)
	# cv2.imwrite('1.jpg', obj1_prepro.image)
	cv2.waitKey(4000)
	cv2.imshow('2.jpg', obj2_prepro.image)
	# cv2.imwrite('2.jpg', obj2_prepro.image)
	cv2.waitKey(4000)