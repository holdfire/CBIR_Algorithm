import os
import cv2
import numpy as np

class BOW:
	def __init__(self, images_dir):
		self.images_dir = images_dir
		# self.MAX_FEATURES = 500
		self.clusterCount = 300
		self.similar_num = 5
		self.DESCRIPTORS = []
		self.vocabulary = []
		self.bow_all = []
		self.idf = []

	def extract_features(self):
		for img in os.listdir(self.images_dir):
			# image reading and to_gray
			image = cv2.imread(os.path.join(self.images_dir, img))
			gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			surf = cv2.xfeatures2d.SURF_create()
			keypoints, descriptors = surf.detectAndCompute(gray_image, None)
			# save each key_point and descriptor of each image
			for i in range(len(descriptors)):
				self.DESCRIPTORS.append(descriptors[i])
		self.DESCRIPTORS = np.array(self.DESCRIPTORS)


	def build_vocabulary(self):
		trainer = cv2.BOWKMeansTrainer(clusterCount=self.clusterCount)
		trainer.add(self.DESCRIPTORS)
		self.vocabulary = trainer.cluster()


	def image_to_unweighted_bow(self, image_path):
		# set the bow image descriptor extractor
		extractor = cv2.xfeatures2d.SURF_create()
		matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
		bowDE = cv2.BOWImgDescriptorExtractor(extractor, matcher)
		bowDE.setVocabulary(self.vocabulary)

		# process the image
		image = cv2.imread(image_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		surf = cv2.xfeatures2d.SURF_create()
		keypoints, descriptors = surf.detectAndCompute(gray_image, None)
		# compute the bow
		image_bow = bowDE.compute(gray_image, keypoints)[0]
		return image_bow

	def all_img_unweighted_bow(self):
		self.extract_features()
		self.build_vocabulary()
		for img in os.listdir(self.images_dir):
			img_path = os.path.join(self.images_dir, img)
			img_bow = self.image_to_unweighted_bow(img_path)
			self.bow_all.append((img_bow, img_path))


	def compute_idf(self):
		idf = np.zeros((self.clusterCount))
		for bow, path in self.bow_all:
			for count in range(self.clusterCount):
				if bow[count]:
					idf[count] += 1
		for i in range(self.clusterCount):
			idf[i] = np.log(len(self.bow_all) / (idf[i] + 1))
		self.idf = idf

	def image_to_tf_idf_weighted_bow(self, image_path):
		image_bow = self.image_to_unweighted_bow(image_path)
		# for i in range(len(image_bow)):
		# 	image_bow[i] = image_bow[i] * image_bow[i] * self.idf[i]
		return image_bow

	def all_img_tf_idf_weighted_bow(self):
		self.all_img_unweighted_bow()
		self.compute_idf()
		# for bow, path in self.bow_all:
		# 	for i in range(len(bow)):
		# 		bow[i] = bow[i] * bow[i] * self.idf[i]
		return self.bow_all


if __name__ == "__main__":
	# testing code
	images_dir = "../data/lib/"
	image_path = "../data/query.jpg"

	obj = BOW(images_dir)
	obj.extract_features()
	obj.build_vocabulary()






