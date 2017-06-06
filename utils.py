import numpy as np
import cv2


def get_keypoint_descriptor_from_images(file_names):
    kp_des = []
    for file_name in file_names:
        img = cv2.imread(file_name, 0)

        if img.empty():
            continue

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        kp_des.append([kp, des])
    return kp_des


def query_is_sub_image(query, train):
    query_kp, query_des = query
    train_kp, train_des = train

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(query_des, train_des, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if len(good) > 10:
        return True
    else:
        return False
