## 1. Introduction
Iput1: a video path (ParaName: video_path)  
Iput2: a directory storing images to be retrived (ParaName: images_to_be_retrieved)  
tools: opencv4  
For each image in input2, we aim to search the most similar image in the video,  
and judge if the two images are the same.  
  
  
## 2. Modules  
video_process: frame the video   
image_preprocess: resize, add mask, delete specified color, smooth region of an image.  
image_extractFeature: extract dHash, pHash, grayHist, hsvHist, bgrHist of an image.  
images_hash: generate the above features of all images.  
image_query: For image to be retrieved, search the most similar image in the video, judge whether they are the same.  
  
  
## 3. Process  
#### STEP1: Frame the video.  
The result will be stored in a new directory called images_from_video  
##### STEP2: Hash all images in images_from_video.  
The reult was stored in a file called images_hash_table.pkl   
#### STEP3: Query images in images_to_be_retrieved.  
Build two new directories called retrieved_regular and retrieved_irregular.  
For each image in images_to_be_retrieved, search the most similar image from the video.  
If they are the same, store it in retrieved_regular directory, else, in the retrieved_irregular directory.  
  
  
## 4. Run this program  
#### On the test data  
run the main.py file.  
#### On your own data  
reset video_path in main.py(line ) to be your own video path,  
build a directory called images_to_be_retrieved which store your own images.  
    (It should be under the same parent directory with video path)  
run the main.py file  
