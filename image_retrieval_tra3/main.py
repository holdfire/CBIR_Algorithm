import os
import time
import pickle
from video_process import video2Images
from images_hash import hashImage
from image_query import queryImage


if __name__ == '__main__':
	# STEP1: process the video to produce pictureszx
	time_start = time.perf_counter()
	video_path = os.path.join(os.getcwd(), "20_e8e90920-dcf0-4837-8581-f22a9c09a1ba.mp4")
	obj_video = video2Images(video_path)
	video_images_dir = obj_video.toImages()
	# video_images_dir = os.path.join(os.path.dirname(video_path), "images_from_video")
	time_end1 = time.perf_counter()
	print("STEP 1: It cost %d s to frame the video !" % (time_end1 - time_start))

	
	# STEP2: hash all images and save
	obj_hash = hashImage()
	obj_hash.hash_batch_images(video_images_dir)
	hash_table_file = obj_hash.save_hash_table(os.path.dirname(video_images_dir))
	time_end2 = time.perf_counter()
	print("STEP 2: It cost %d s to hash all referenced images !" % (time_end2 - time_end1))


	# STEP3: first query of each image in the imgs_to_be_retrieved directory
	with open(hash_table_file, "rb+") as f:
		hash_table = pickle.load(f)
	# query each image in the directory
	images_to_be_retrieved_dir = os.path.join(os.path.dirname(video_path), 'images_to_be_retrieved')
	images = os.listdir(images_to_be_retrieved_dir)
	for image_name in images:
		image_path = os.path.join(images_to_be_retrieved_dir, image_name)
		obj_query = queryImage(image_path)
		obj_query.query_single_image(hash_table)
	time_end3 = time.perf_counter()
	# save query log, and show the query result
	dirname = os.path.dirname(video_path)
	queryImage.cls_save_query_result_log(dirname)
	failed_image_path = queryImage.cls_classify_queried_image(dirname)
	print("STEP 3: It cost %d s to retrieve  %d images !" % (time_end3 - time_end2, len(images)))









