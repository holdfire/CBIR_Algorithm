import cv2
import os
import time
import numpy as np


class ColorHist:
    '''
    Introduction: Using the following methods, we can get a histogram description of a given image.
    hsv hist is suggested.
    gray hist, rgb hist are not suggested, as they are not so variant.
    '''
    def __init__(self, image_path, new_size=(128, 128)):
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_CUBIC)

    def hsv_hist(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = np.split(hsv_image, 3, axis=2)
        h = h.reshape((h.shape[0], h.shape[1]))
        s = s.reshape((s.shape[0], s.shape[1]))
        v = v.reshape((v.shape[0], v.shape[1]))
        # 对hsv色彩空间的h, s, v三个通道分别进行映射
        self.hsv_mapping(h, s, v)
        hsv_matrix = np.array(9 * h + 3 * s + v, dtype=np.uint8)
        mapped_hsv_hist = cv2.calcHist([hsv_matrix], [0], None, [72], [0, 72]).reshape((1, -1))[0]
        return mapped_hsv_hist

    def hsv_mapping(self, h, s, v):
        # 对hsv图像先做预处理，接近黑色的均设为黑色
        h[v<38] = 0
        s[v<38] = 0
        v[v<38] = 0
        # 接近白色的均设为白色
        h[(s<26) & (v>204)] = 0
        s[(s<26) & (v>204)] = 0
        v[(s<26) & (v>204)] = 255
        # 将色调空间hue由[0, 180)这个区间映射到[0, 7]
        h[((0 <= h) & (h <= 10)) | (158 <= h) & (h <= 180)] = 0
        h[((11 <= h) & (h <= 20))] = 1
        h[((21 <= h) & (h <= 37))] = 2
        h[((38 <= h) & (h <= 77))] = 3
        h[((78 <= h) & (h <= 95))] = 4
        h[((96 <= h) & (h <= 135))] = 5
        h[((136 <= h) & (h <= 147))] = 6
        h[((148 <= h) & (h <= 157))] = 7
        # 将饱和度空间saturation由[0, 255)这个区间映射到[0, 2]
        s[((0 <= s) & (s <= 51))] = 0
        s[((52 <= s) & (s <= 178))] = 1
        s[((179 <= s) & (s <= 256))] = 2
        # 对明度空间value由[0, 255)这个区间映射到[0, 2]
        v[((0 <= v) & (v <= 51))] = 0
        v[((52 <= v) & (v <= 178))] = 1
        v[((179 <= v) & (v <= 256))] = 2


if __name__ == "__main__":
    path = "../data/imageLib/"

    start = time.perf_counter()
    os.chdir(path)
    for image in os.listdir(os.getcwd()):
        image = cv2.imread(image)
        image = cv2.resize(image,(128,128))
        obj = ColorHist(image)
        vec = obj.hsv_hist()
        print('hi')
    end = time.perf_counter()
    print(end-start)

