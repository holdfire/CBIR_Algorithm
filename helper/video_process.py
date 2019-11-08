import cv2
import os


def video2Images(video_path, frame_interval=3, resize=False, new_size=(128, 128)):
	# build a directory to store images
	images_dir = os.path.join(os.path.dirname(video_path), 'images_dir')
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
		image_path = os.path.join(images_dir, os.path.basename(video_path) + "_" + str(count) + ".jpg")
		cv2.imwrite(image_path, frame)
	return images_dir


if __name__ == "__main__":
	pass
