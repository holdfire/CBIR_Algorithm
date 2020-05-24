import cv2
import numpy as np

def cmp_vector(vec1, vec2):
    return cv2.compareHist(vec1, vec2, cv2.HISTCMP_INTERSECT) / np.sum(vec1)



if __name__ == "__main__":

    vec1 = np.random.randint(0,10,100)
    vec2 = (vec1 + np.random.randint(0,3,1))
    score = cmp_vector(vec1, vec2)
    print(score)

