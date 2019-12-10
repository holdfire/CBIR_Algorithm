import os
import pickle
import csv
import cv2
import faiss
import numpy as np

def faiss_search(x_query, x_base, top_k = 4):
	'''
	using faiss module, search the top_k similar vectors of x_query in x_base.
	:param x_query: query vector, a list, eg: [1,1,1,1]
	:param x_base:  database, numpy.ndarray, eg[[1,1,1,1], [2,2,2,2], [3,3,3,3]]
	:param top_k:   we want to see top_k nearest neighbors
	:return: an integer(from zero): represents the index of the best matching vector. -1 represents not found
	'''
	if not x_query.shape:
		raise Exception("The input query vector is empty!")
	if x_query.shape[1] != x_base.shape[1]:
		raise Exception("The query vector does not match the base vector")
	# the number and dimension of the query numpy.ndarray
	nb, dim = x_query.shape
	# Build inverted indexing. brute force search by comparing euclidean distance
	# if you have a GPU, use faiss.GPUIndexFlatL2()
	index = faiss.IndexFlatL2(dim)
	# add vectors to the index
	index.add(x_base)
	# convert x_query into numpy.ndarray type
	# x_q = np.array([x_query])
	Distance, Index = index.search(x_query, top_k)

	best_index = Index[:,0]
	best_index = np.squeeze(best_index)
	# best_score = cmp_vector(x_query, x_base[best_index])
	# if best_score >= score_threshold:
	# 	return best_index
	return best_index


def sequential_search(x_query, x_base, score_threshold = 0.8):
	'''
	:param x_query: query vector,a list, eg: [1,1,1,1]
	:param x_base:  database vector, numpy.ndarray, eg[[1,1,1,1], [2,2,2,2], [3,3,3,3]]
	:param score_threshold: the threshold to judge whether two vectors match
	:return: -1 represents not found, 0 represent the first one, a positive integer represents the index of the best matching vector
	'''
	if not len(x_query):
			raise Exception("The input query vector is empty!")
	if len(x_query) != x_base.shape[1]:
			raise Exception("The query vector does not match the base vector")
	best_index = -1
	best_score = 0
	index = 0
	for x in x_base:
			score = cmp_vector(x, x_query)
			if score > best_score:
					best_score = score
					best_index = index
			index += 1
	if best_score >= score_threshold:
			return best_index
	return -1

def cmp_vector(vec1, vec2):
	return cv2.compareHist(vec1, vec2, cv2.HISTCMP_INTERSECT) / np.sum(vec1)



if __name__ == "__main__":

	os.chdir("../data/online/")
	with open("hsv_hist.pkl", "rb+") as f1:
		x_base = np.array(pickle.loads(f1))
	with open("queries_hsv_hist.pkl", "rb+") as f2:
		x_query = np.array(pickle.loads(f2))
	best_index = faiss_search(x_query, x_base)


	with open("image_path.pkl", "rb+") as f3:
		y_base = np.array(pickle.loads(f3))[best_index]
	with open("queries_image_path.pkl", "rb+") as f4:
		y_query = pickle.loads(f4)

	headers = ["query_image", "matched_image"]
	with open(os.path.join('../', "result.csv"), "wb+", newline = '') as f:
		f_csv = csv.writer(f, headers)
		for i in range(len(y_query)):
			row = []
			row.append(y_query[i])
			row.append(y_base[i])
			f_csv.writerow(row)








