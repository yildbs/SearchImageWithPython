'''
Code below came from opencv example (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
Because of difference between opencv 3-beta and opencv 3.2.0, i modified the code
'''
import numpy as np
import cv2

#img1 = cv2.imread('../images/box.png',0)          # queryImage
#img1 = cv2.imread('../images/human_upper_body.png',0)          # queryImage
#img2 = cv2.imread('../images/box_in_scene.png',0) # trainImage
img1 = cv2.imread('/home/yildbs/Data/INRIA/SomeSample/crop_org_2_1.png',0)          # queryImage
img2 = cv2.imread('/home/yildbs/Data/INRIA/SomeSample/org_1.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = np.array([])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, flags=2)

cv2.imshow('aaa', img3)
cv2.waitKey(0)
