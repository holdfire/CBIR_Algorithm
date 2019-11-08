import cv2
import numpy as np


class LocalFeature():
    def __init__(self,image_path):
        self.image = cv2.imread(image_path)

    def get_features(self, method = "ORB", MAX_FEATURES = 6000):
        '''
        You can get keypoints and descriptors of image, and choose different kinds of feature.
        For SIFT and SURF feature, OpenCV no longer supports api.
        You can uninstall OpenCV, and install opencv-contrib-python3.4.2 in python
        :return: kepoints, descriptors
        '''
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if method == "SIFT":
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        elif method == "SURF":
            surf = cv2.xfeatures2d.SURF_create()
            keypoints, descriptors = surf.detectAndCompute(gray_image, None)
        elif method == "ORB":
            orb = cv2.ORB_create(MAX_FEATURES)
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        else:
            raise Exception("Method Not Found!")
        return keypoints, descriptors



class alignImages():
    def __init__(self, image_path1, image_path2):
        self.image_path1 = image_path1
        self.image_path2 = image_path2

    def matchImages(self, GOOD_MATCH_PERCENT=0.2):
        # get keypoints and descriptors of two images
        local1 = LocalFeature(self.image_path1)
        keypoints1, descriptors1 = local1.get_features()
        local2 = LocalFeature(self.image_path2)
        keypoints2, descriptors2 =  local2.get_features()

        # the matcher rely on the descriptors type
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
        matches = matcher.match(descriptors1, descriptors2, None)
        # Sort matches by distance
        matches.sort(key=lambda x: x.distance, reverse=False)
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        image1 = cv2.imread(self.image_path1)
        image2 = cv2.imread(self.image_path2)
        imMatches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography by RANSAC
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        # Use homography, image1为被对齐的图片
        height, width, channels = image2.shape
        image1_aligned = cv2.warpPerspective(image1, h, (width, height))
        return image1_aligned


if  __name__ == "__main__":
    # testing code
    image_path1 = '../data/query.jpg'
    image_path2 = '../data/lib/cocacola_1.jpg'
    obj = alignImages(image_path1, image_path2)
    obj.matchImages()



