import os
import cv2
import numpy as np


img = cv2.imread("../data/online/test/002.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg", img)

width, height = gray.shape


# Canny edge detection
edge = cv2.Canny(gray, 20, 50)
# cv2.imwrite("edge_opencv_canny.jpg", edge)
# cv2.waitKey(10)

# Hough transformaiton to detect lines
lines = cv2.HoughLinesP(edge, 1, np.pi/180, 100)
lines = np.squeeze(lines)
for i in range(len(lines)):
    line = lines[i]
    length = np.sqrt((line[0] - line[2])**2 + (line[1] - line[3])**2)
    if (width * 0.1 < length) & (height * 0.1 < length):
        img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,0,255), 3, 8)
cv2.imwrite("opencv_hough.jpg", img)
cv2.waitKey(10)


