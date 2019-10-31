import cv2
import numpy as np

class Hashing:
    def __init__(self, image):
        self.image = image

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

    def pHash(self, hash_size = (32, 32)):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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


