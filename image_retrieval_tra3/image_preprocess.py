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

	def clip_rect_image(self, rect_prop=(0.25, 0.25)):
		# clipped image size >= original image size * rect_prop
		return self.image


if __name__ == "__main__":
	path = "C:\\Users\\Dell\\Desktop\\omi_ori_data\\omi_ori_data\\292_5cbd9da23d4b4376a358ea68a79e2f96\\292_taskDown\\images_to_be_retrieved\\1.jpg"
	image = cv2.imread(path)
	obj = preprocessImage(image)
	obj.resize_image()
	obj.mask_image()
	# obj.delete_specified_color()
	# obj.region_smooth()

	cv2.imshow('current.jpg', obj.image)
	cv2.waitKey(0)






