import cv2
import os
import numpy as np


class RectDetector():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def canny(self):
        # 去噪滤波器：网页截图可以不做滤波，相机拍摄图片可以考虑做滤波处理；
        # 中值滤波是一种非线性滤波，它能在滤除噪声的同时很好的保持图像边缘，优先考虑使用
        # 双边高斯滤波cv2.bilateralFilter()，在滤波的同时能保证一定的边缘信息，但运行较慢，也可考虑使用；
        # 高斯滤波cv2.GaussianBlur()、均值滤波cv2.blur()会模糊边缘，不建议使用。
        image = cv2.medianBlur(self.gray_image, (3,3))
        # 边缘提取滤波器：

    def detection(self, threshold=240, maxval=1, type=cv2.THRESH_BINARY):
        # 对原图依次进行灰度化-->边缘检测

        binary_image = cv2.threshold(self.gray_image, threshold, maxval, type)[1].astype(np.uint8)
        # 检测轮廓-->计算各轮廓包围的面积-->找到面积前2的子图-->排除原图后，返回面积最大的子图在轮廓数组中的索引
        contours = np.array(cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1])
        contours_area = []
        for i in range(len(contours)):
            contours_area.append(cv2.contourArea(contours[i]))
        top2_index = np.argsort(contours_area)[: :-1][:2]
        # 注意可能没有找到轮廓，这时候只有一个子图即原图，直接返回原图
        if len(top2_index) < 2:
            return self.image
        # 如果检测到的第一大的子图是原图，则取第二大的子图；否则取第一大的子图
        if cv2.contourArea(contours[top2_index[0]]) > self.image.shape[0] * self.image.shape[1]:
            max_index = top2_index[1]
        else:
            max_index = top2_index[0]
        # cv2.drawContours(self.image, contours[top2_index], 0, (0,0,255), 3)

        # 对该面积最大子图轮廓上的点进行排序-->找到左上角点(x1, y1)和右下角点(x2, y2)-->返回包围该子图的一个矩形子图
        max_counter = np.squeeze(contours[max_index])
        counter_point_rank = np.sum(max_counter, axis = 1)
        (x1, y1) = max_counter[np.argmin(counter_point_rank)]
        (x2, y2) = max_counter[np.argmax(counter_point_rank)]
        (y_max, x_max) = np.array(self.image.shape[:2]) - 1
        self.image = self.image[(y1 if y1 >=0 else 0):(y2 if y2<=y_max else y_max), (x1 if x1 >=0 else 0):(x2 if x2<=x_max else x_max)]
        return self.image



if __name__ == "__main__":
    os.chdir("../data/test")
    for image_path in os.listdir("./"):
        obj = RectDetector(image_path)
        new_image = obj.detection()
        cv2.imwrite("processed" + image_path, new_image)







