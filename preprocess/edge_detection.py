import cv2
import numpy as np


class Edge():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def getEdge(self):
        gray_image = cv2.cvtColor(self.image, cv2.BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5,5), 0)
        # 后面两个参数为低阈值、高阈值
        canny = cv2.Canny(gray_image, 50, 150)
        cv2.imshow("canny",canny)
        cv2.waitKey()







if __name__ == "__main__":
    image_path =