import cv2
import faiss
import numpy as np

def faiss_search(x_query, x_base, top_k = 4, score_threshold = 0.8):
	'''
	using faiss module, search the top_k similar vectors of x_query in x_base.
	:param x_query: query vector, a list, eg: [1,1,1,1]
	:param x_base:  database, numpy.ndarray, eg[[1,1,1,1], [2,2,2,2], [3,3,3,3]]
	:param top_k:   we want to see top_k nearest neighbors
	:return: an integer(from zero): represents the index of the best matching vector. -1 represents not found
	'''
	if not len(x_query):
		raise Exception("The input query vector is empty!")
	if len(x_query) != x_base.shape[1]:
		raise Exception("The query vector does not match the base vector")
	dim = len(x_query)  # the dimension of a single query vector
	index = faiss.IndexFlatL2(dim)  # build the index
	index.add(x_base)  # add vectors to the index
	# convert x_query into numpy.ndarray type
	x_q = np.array([x_query])
	Distance, Index = index.search(x_q, top_k)
	best_index = Index[0][0]
	best_score = cmp_vector(x_query, x_base[best_index])
	if best_score >= score_threshold:
		return best_index
	return -1


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
	d = 64
	nb = 100000
	nq = 1000
	np.random.seed(1234)
	xb = np.random.random((nb, d)).astype('float32')
	xb[:, 0] += np.arange(nb) / 1000.
	xq = np.random.random((nq, d)).astype('float32')
	xq[:, 0] += np.arange(nq) / 1000.

	print(faiss_search(xb[1], xb, 4))
	print(sequential_search(xb[4], xb))







