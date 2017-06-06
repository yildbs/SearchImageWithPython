import glob
import utils
import cv2

if __name__ == "__main__":
    train_file_list = glob.glob("../Data/INRIA/SomeSample/org*")
    query_file_list = glob.glob("../Data/INRIA/SomeSample/crop*")

    train_kp_des = utils.get_keypoint_descriptor_from_images(train_file_list)
    query_kp_des = utils.get_keypoint_descriptor_from_images(query_file_list)

    for query_zip in zip(query_kp_des, query_file_list):
        query, query_file_name = query_zip[0], query_zip[1]

        for train_zip in zip(train_kp_des, train_file_list):
            train, train_file_name = train_zip[0], train_zip[1]

            if utils.query_is_sub_image(query, train) is True:
                query_image = cv2.imread(query_file_name)
                train_image = cv2.imread(train_file_name)
                cv2.imshow('query', query_image)
                cv2.imshow('train', train_image)
                cv2.waitKey(0)
