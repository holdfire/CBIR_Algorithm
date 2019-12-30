import cv2
import os


def video2Images(video_path, frame_interval=3, resize=False, new_size=(128, 128)):
	# build a directory to store images
	images_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'img_lib')
	if not os.path.exists(images_dir):
		os.makedirs(images_dir)

	video = cv2.VideoCapture(video_path)
	count = 0
	rval = video.isOpened()
	# read the video by frames
	while rval:
		count = count + 1
		rval, frame = video.read()
		if count % frame_interval != 0:
			continue
		if resize:
			frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
		# 最后一帧为空，需要丢弃掉
		if frame is not None:
			image_name = os.path.basename(video_path).split('-')[0] + "_" + str(count) + ".png"
			image_path = os.path.join(images_dir, image_name)
			cv2.imwrite(image_path, frame)
	return images_dir

def video2keyFrames(video_path):
	'''
	Extract key frames from a video.
	:param video_path:
	:return:
	'''
	pass



if __name__ == "__main__":
	# 将工作目录切换到存放视频的目录，注意：视频广告的名称尽量为合法变量名称字符
	os.chdir("../data/online/videos")
	for video in os.listdir(os.getcwd()):
		video_path = os.path.join(os.getcwd(), video)
		video2Images(video_path)


