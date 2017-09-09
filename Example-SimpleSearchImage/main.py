import glob
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


if __name__ == "__main__":
    train_file_list = glob.glob("../Data/INRIA/SomeSample/org*")
    query_file_list = glob.glob("../Data/INRIA/SomeSample/crop*")

    train_kp_des = get_keypoint_descriptor_from_images(train_file_list)
    query_kp_des = get_keypoint_descriptor_from_images(query_file_list)

    for query_zip in zip(query_kp_des, query_file_list):
        query, query_file_name = query_zip[0], query_zip[1]

        for train_zip in zip(train_kp_des, train_file_list):
            train, train_file_name = train_zip[0], train_zip[1]

            if query_is_sub_image(query, train) is True:
                query_image = cv2.imread(query_file_name)
                train_image = cv2.imread(train_file_name)
                cv2.imshow('query', query_image)
                cv2.imshow('train', train_image)
                cv2.waitKey(0)
