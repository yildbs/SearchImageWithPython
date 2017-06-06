'''
Code below came from opencv example (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
Because of difference between opencv 3-beta and opencv 3.2.0, i modified the code
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../images/box.png',0)          # queryImage
img2 = cv2.imread('../images/box_in_scene.png',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches
img3 = np.array([])
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img3, flags=2)

cv2.imshow('img3', img3)
cv2.waitKeyEx(0)