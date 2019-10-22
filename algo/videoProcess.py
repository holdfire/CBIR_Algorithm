import cv2
import os
import pickle

class video2Images:
	def __init__(self):
		self.video_dir = ".//..//tmp//video"
		self.exist_video = ".//..//data//exist_video.pkl"
		self.imgLib = ".//..//imgLib"
		return

	def get_new_videos(self):
		# list videos newly uploaded by users
		if not os.path.exists(self.video_dir):
			os.mkdir(self.video_dir)
		videos = os.listdir(self.video_dir)

		# check the format of videos
		valid_videos = []
		for item in videos:
			if item.split('.')[-1] == "mp4":
				valid_videos.append(item)

		# check whether the video is new, but not existed
		new_video = []
		with open(self.exist_video, 'rb') as f:
			try:
				exist_video = pickle.load(f)
			except EOFError:
				exist_video = []
		###########################  delete the next line when using ########################
		exist_video = []
		#####################################################################################
		for item in valid_videos:
			if not item in exist_video:
				new_video.append(item)
				exist_video.append(item)

		# add new video to the exist_video file
		with open(self.exist_video, 'wb+') as f:
			pickle.dump(exist_video, f)
		self.new_video = new_video
		return self.new_video

	def single_video_to_images(self,video_name, frame_interval = 3, resize = False, new_size = (256, 256)):
		img_dir = os.path.join(self.imgLib, video_name.split('.')[0])
		if not os.path.exists(img_dir):
			os.mkdir(img_dir)
		video = cv2.VideoCapture(os.path.join(self.video_dir, video_name))
		count = 0
		rval = video.isOpened()
		# read the video by frames
		while rval:
			count = count + 1
			rval, frame = video.read()
			if count % frame_interval != 0:
				continue
			if rval:
				image = frame
			if resize:
				image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
			image_path = os.path.join(img_dir, video_name.split('.')[0] + "_" + str(count) + ".jpg")
			cv2.imwrite(image_path, image)
		return img_dir

	def batch_video_to_image(self):
		for video in self.new_video:
			self.single_video_to_images(video)
		return

	def remove_videos(self):
		os.chdir(self.video_dir)
		for item in os.listdir(os.getcwd()):
			os.remove(item)
		return


if __name__ == '__main__':
	obj = video2Images()
	obj.get_new_videos()
	obj.batch_video_to_image()
	# obj.remove_videos()




