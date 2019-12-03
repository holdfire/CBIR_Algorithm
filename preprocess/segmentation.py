import cv2
import os
import numpy as np


class SegImg():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def seg_by_contours(self, padding=3, padding_value=[255, 255, 255], threshold=240, maxval=1, type=cv2.THRESH_BINARY):
        '''
        1. 广告图片质量非常好，没有噪点。不应该做高斯滤波，做了效果反而不好；
        2.  本算法能很好地切割由黑色和白色组成的边缘
        :param padding: 在原图（指self.image）的4个边界做填充，目的是检测出原图的4个边界；
        :param padding_value: 原图边界填充的像素点，默认填充白色像素；
        :param threshold: 对原图做二值化处理的阈值，默认去白边；如果去黑边则阈值应该设置较大
        :param maxval: 默认当像素点的值超过上面阈值时，设置为maxval，否则设置为0；
        :param type: 默认的二值化方法；
        :return: 原图的最大矩形子图
        '''
        # 对原图依次进行边缘填充，灰度化，二值化处理
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=padding_value)
        gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.threshold(gray_image, threshold, maxval, type)[1].astype(np.uint8)

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

        # 对该面积最大子图轮廓上的点进行排序-->找到左上角点(x1, y1)和右下角点(x2, y2)-->返回包围该子图的一个矩形子图
        max_counter = np.squeeze(contours[max_index])
        counter_point_rank = np.sum(max_counter, axis = 1)
        (x1, y1) = max_counter[np.argmin(counter_point_rank)] - padding + 1
        (x2, y2) = max_counter[np.argmax(counter_point_rank)] - padding - 1
        (y_max, x_max) = np.array(self.image.shape[:2]) - 1
        self.image = self.image[(y1 if y1 >=0 else 0):(y2 if y2<=y_max else y_max), (x1 if x1 >=0 else 0):(x2 if x2<=x_max else x_max)]
        return self.image



if __name__ == "__main__":
    # image_path = "../data/test/query.png"
    # obj = SegImg(image_path)
    # #obj.seg_by_contours()
    # new_image = obj.seg_by_contours(padding_value=[0, 0, 0], threshold=50, type=cv2.THRESH_BINARY_INV)
    # os.chdir(os.path.dirname(image_path))
    # cv2.imwrite("processed_image.png", new_image)

    os.chdir("../data/queries")
    for img in os.listdir(os.getcwd()):
        obj = SegImg(img)
        obj.seg_by_contours()
        new_image = obj.seg_by_contours(padding_value=[0, 0, 0], threshold=10, type=cv2.THRESH_BINARY_INV)
        image_path = os.path.join("../queries_processed",img)
        cv2.imwrite(image_path, new_image)






