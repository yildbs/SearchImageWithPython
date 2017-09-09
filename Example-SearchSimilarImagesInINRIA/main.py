import glob
import numpy as np
import cv2
import random
import os
import utils.featureextractor


if __name__=="__main__":
    print("Search similar images in INRIA")
    print("Currently working directory : " , os.getcwd())

    sift = utils.featureextractor.SiftFeatureExtractor()

    feature_database = []
    for index, path in zip(range(999), sorted(glob.glob("../INRIA_Train_original/pos/*.png"))):
        feature_database.append(sift.get_image_feature_with_filename(path))

    targets = []
    for index, path in zip(range(999), sorted(glob.glob("../INRIA_Train_160x96/pos/*.png"))):
        targets.append(sift.get_image_feature_with_filename(path))

    for target in targets:
        found = False
        matched = []
        for saved in feature_database:
            associated = target.compare_with(saved)
            if associated:
                found = True
                matched.append(saved)
        if not found:
            continue

        target_image = cv2.imread(target.get_filename())
        cv2.imshow("target_image", target_image)
        for index, match in zip(range(len(matched)), matched):
            match_image = cv2.imread(match.get_filename())
            cv2.imshow("match_image_"+str(index), match_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
