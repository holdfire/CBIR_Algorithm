from __future__ import print_function
import cv2
import numpy as np

# rotate and resize im1 to be aligned with im2
def alignImages(im1, im2):

    # Convert adds to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

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
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
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
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h



if __name__ == '__main__':

    MAX_FEATURES = 6000
    GOOD_MATCH_PERCENT = 0.2

    refFilename = "D:\\APPData\\PythonProject\\data\\adds7\\ref2.jpg"
    imFilename = "D:\\APPData\\PythonProject\\data\\adds7\\regular.jpg"

    # Read reference image
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    # Read image to be aligned
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning adds ...")
    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)

