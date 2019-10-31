import cv2
import numpy as np

class Hist:
    def __init__(self, image):
        self.image = image

    def hsv_hist(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = np.split(hsv_image, 3, axis=2)
        h = h.reshape((h.shape[0], h.shape[1]))
        s = s.reshape((s.shape[0], s.shape[1]))
        v = v.reshape((v.shape[0], v.shape[1]))
        hsv_matrix = np.ones(h.shape)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                # 对hsv图像先做预处理，接近黑色的均设为黑色，接近白色的均设为白色
                if (v[i][j] < 38):
                    h[i][j] = 0
                    s[i][j] = 0
                    v[i][j] = 0
                elif ((s[i][j] < 26) & (v[i][j] > 204)):
                    h[i][j] = 0
                    s[i][j] = 0
                    v[i][j] = 255
                # 借助上述hsv映射关系，将hsv色彩空间转换为[8, 3, 3]维的空间
                h[i][j] = Hist.Hue_list[h[i][j]]
                s[i][j] = Hist.Saturation_list[s[i][j]]
                v[i][j] = Hist.Value_list[v[i][j]]
                hsv_matrix[i][j] = 9 * h[i][j] + 3 * s[i][j] + v[i][j]
                hsv_matrix = np.array(hsv_matrix, dtype=np.uint8)
        recoded_hsvHist = cv2.calcHist([hsv_matrix], [0], None, [72], [0, 72]).reshape((1, -1))[0]
        return recoded_hsvHist

    # assistant variable or method
    Hue_list = []
    Saturation_list = []
    Value_list = []

    @classmethod
    def hsv_map(cls):
        cls.Hue_list = np.arange(180)
        cls.Saturation_list = np.arange(256)
        cls.Value_list = np.arange(256)
        for i in range(len(cls.Hue_list)):
            if ((0 <= i) & (i <= 10)) | (158 <= i) & (i <= 180):
                cls.Hue_list[i] = 0
            elif ((11 <= i) & (i <= 20)):
                cls.Hue_list[i] = 1
            elif ((21 <= i) & (i <= 37)):
                cls.Hue_list[i] = 2
            elif ((38 <= i) & (i <= 77)):
                cls.Hue_list[i] = 3
            elif ((78 <= i) & (i <= 95)):
                cls.Hue_list[i] = 4
            elif ((96 <= i) & (i <= 135)):
                cls.Hue_list[i] = 5
            elif ((136 <= i) & (i <= 147)):
                cls.Hue_list[i] = 6
            elif ((148 <= i) & (i <= 157)):
                cls.Hue_list[i] = 7
        for i in range(len(cls.Saturation_list)):
            if ((0 <= i) & (i <= 51)):
                cls.Saturation_list[i] = 0
            elif ((52 <= i) & (i <= 178)):
                cls.Saturation_list[i] = 1
            elif ((179 <= i) & (i <= 256)):
                cls.Saturation_list[i] = 2
        for i in range(len(cls.Value_list)):
            if ((0 <= i) & (i <= 51)):
                cls.Value_list[i] = 0
            elif ((52 <= i) & (i <= 178)):
                cls.Value_list[i] = 1
            elif ((179 <= i) & (i <= 256)):
                cls.Value_list[i] = 2
        return cls.Hue_list, cls.Saturation_list, cls.Value_list

if __name__ == "__main__":
    image_path = "../data/query.jpg"
    image = cv2.imread(image_path)
    Hist.hsv_map()
    obj = Hist(image)
    vec = obj.hsv_hist()
    print(vec)
