import cv2
import numpy as np


class preprocessImage:
	def __init__(self, image):
		self.image = image

	def resize_image(self, new_size = (128, 128)):
		self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_CUBIC)
		return self.image

	def mask_image(self, region=(28, 56, 1, 83)):
		# build the mask matrix
		mask = np.ones(self.image.shape[:2], np.uint8)
		mask[region[0]:region[1], region[2]:region[3]] = 0
		# split the bgr channels, the do mask the black area of each channel
		channels = cv2.split(self.image)
		b = cv2.bitwise_and(channels[0], channels[0], mask=mask)
		g = cv2.bitwise_and(channels[1], channels[1], mask=mask)
		r = cv2.bitwise_and(channels[2], channels[2], mask=mask)
		# merge the masked bgr channels
		self.image = cv2.merge([b, g, r])
		return self.image

	def delete_specified_color(self, hsv_range=(35,77,43,256,46,256), region=(28, 56, 1, 83)):
		# delete specified color, bgr = ()
		hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
		for i in range(region[1] - region[0]):
			for j in range(region[3] - region[2]):
				x = i + region[0]
				y = j + region[2]
				if ((hsv_range[0] < hsv_image[x][y][0]) & (hsv_image[x][y][0] < hsv_range[1]) & \
					(hsv_range[2] < hsv_image[x][y][1]) & (hsv_image[x][y][1] < hsv_range[3]) & \
					(hsv_range[4] < hsv_image[x][y][2]) & (hsv_image[x][y][2] < hsv_range[5])):
					hsv_image[x][y] = [0,0,0]
					hsv_image[x-1][y] = [0, 0, 0]
					hsv_image[x+1][y] = [0, 0, 0]
					hsv_image[x][y-1] = [0, 0, 0]
					hsv_image[x][y+1] = [0, 0, 0]
		self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
		return self.image

	def region_smooth(self, region=(28, 56, 1, 83)):
		# smooth the hsv pixel in the region
		for i in range(region[1] - region[0]):
			for j in range(region[3] - region[2]):
				x = i + region[0]
				y = j + region[2]
				if np.mean(self.image[x][y]) == 0:
					neighbor = []
					if np.mean(self.image[x+1][y]) != 0:
						neighbor.append(self.image[x+1][y])
					if np.mean(self.image[x-1][y]) != 0:
						neighbor.append(self.image[x-1][y])
					if np.mean(self.image[x][y-1]) != 0:
						neighbor.append(self.image[x][y-1])
					if np.mean(self.image[x][y+1]) != 0:
						neighbor.append(self.image[x][y+1])
					self.image[x][y] = np.mean(neighbor)
		return self.image

	def segment_image(self):
		img = self.image
		mask = np.zeros(img.shape[:2], np.uint8)
		bgdModel = np.zeros((1, 65), np.float64)
		fgdModel = np.zeros((1, 65), np.float64)
		rect = (0, 0,img.shape[0], img.shape[1])
		cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
		img = img * mask2[:, : , np.newaxis]
		return img

	def detect_edge(self):
		img = self.image
		gray_image = cv2.medianBlur(img, 5)
		gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

		canny_image = cv2.Canny(gray_image, 60, 150)
		lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 80, 30, 10)
		print(lines.shape)
		for i in range(len(lines)):
			startx = lines[i][0][0]
			starty = lines[i][0][1]
			endx = lines[i][0][2]
			endy = lines[i][0][3]
			self.image = cv2.line(self.image, (startx, starty), (endx, endy), (255,0,0) )
		return self.image

	def detect_rect(self):
		img = self.image
		img = cv2.medianBlur(img, 5)
		b, g, r = cv2.split(img)



if __name__ == "__main__":
	path = "C:\\Users\\Dell\\Desktop\\omi_ori_data\\1.jpg"
	image = cv2.imread(path)
	obj = preprocessImage(image)
	#result = obj.segment_image()
	result = obj.detect_edge()
	# obj.resize_image()
	# obj.mask_image()

	cv2.imshow('current.jpg', result)
	cv2.waitKey(0)






