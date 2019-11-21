import cv2
import os
import numpy as np


class SegByContour():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)


    def segImage(self, top_k=3, padding=3, threshold=250):
        '''
        Attention：广告图片质量非常好，没有噪点。不应该做高斯滤波，做了效果反而不好。可以自己设置滤波器做优化
        :return: 原图像的top_k个子图的轮廓，其中子图为矩形
        '''
        # 将原图的4个边缘填充白色像素，不做padding的话，原图的边界检测不出来
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value = [255,255,255])
        # 再处理成灰度图
        gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        # 对灰度图做二值化处理，便于找到轮廓
        binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        # 寻找图像中块的轮廓，返回一个list，其元素为各个轮廓---->修改为numpy.ndarray类型
        contours = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = np.array(contours)
        # 计算各轮廓所包围块的面积
        contours_area = []
        for i in range(len(contours)):
            contours_area.append(cv2.contourArea(contours[i]))
        # 找到面积最大的前k+1个轮廓的索引，为什么是k+1呢？看下面的注释
        top_k_index = np.argsort(contours_area)[: :-1][:top_k+1]
        # 如果子图中包含原图，应该将其去掉
        if cv2.contourArea(contours[top_k_index[0]]) > self.image.shape[0] * self.image.shape[1]:
            top_k_index = top_k_index[1:]
        else:
            top_k_index = top_k_index[:top_k]
        self.top_k_contours = contours[top_k_index]
        cv2.drawContours(padded_image, self.top_k_contours, -1, (0,0,255), 2)
        self.padded_image = np.array(padded_image)
        return self.padded_image
        # cv2.imshow("contour_padded_image.png", self.padded_image)
        # cv2.waitKey(5)
        # return self.top_k_contours


    # def getSubImage(self):
    #     for i in range(self.top_k_contours)


if __name__ == "__main__":
    # image_path = "../data/test/query3.png"
    # obj = SegByContour(image_path)
    # new_image = obj.segImage()
    # os.chdir(os.path.dirname(image_path))
    # cv2.imwrite("processed_image.png", new_image)

    os.chdir("../data/lib2")
    for img in os.listdir(os.getcwd()):
        obj = SegByContour(img)
        new_image = obj.segImage()
        image_path = os.path.join("../lib2_processed",img)
        cv2.imwrite(image_path, new_image)






