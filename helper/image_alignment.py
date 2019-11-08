from __future__ import print_function
import cv2
import numpy as np

def alignImages(image1, image2):
    '''
    rotate and resize im1 to be aligned with im2
    :param image1:
    :param image2:
    :return:
    '''
    # Convert adds to grayscale
    im1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect ORB feature and compute descriptors.
    # keypoints shape: (MAX_FEATURES, 1)
    # descriptors shape: (MAX_FEATURES, 32)
    # orb = cv2.ORB_create(MAX_FEATURES)
    # keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)


    # Match feature.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    imMatches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)


    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt


    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = image2.shape
    im1Reg = cv2.warpPerspective(image1, h, (width, height))

    return im1Reg, h



if __name__ == '__main__':

    MAX_FEATURES = 6000
    GOOD_MATCH_PERCENT = 0.2


