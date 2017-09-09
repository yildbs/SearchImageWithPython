import cv2


class ImageFeature():
    def __init__(self, filename, image, keypoint, descriptor):
        self.filename = filename
        self.image = image
        self.keypoint = keypoint
        self.descriptor = descriptor

    def get_keypoint(self):
        return self.keypoint

    def get_image(self):
        return self.image

    def get_filename(self):
        return self.filename

    def get_descriptor(self):
        return self.descriptor

    def compare_with(self, image_feature):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.descriptor, image_feature.get_descriptor(), k=2)

        # Apply ratio test
        good_match = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_match.append([m])

        return len(good_match) > 10


class SiftFeatureExtractor():
    def __init__(self):
        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()

    def get_image_feature_with_filelist(self, filelist):
        images = []
        for filename in filelist:
            image = cv2.imread(filename)

            # Find the keypoints and descriptors with SIFT
            keypoint, descriptor = self.sift.detectAndCompute(image, None)
            images.append(ImageFeature(filename, image, keypoint, descriptor))
        return images

    def get_image_feature_with_filename(self, filename):
        image = cv2.imread(filename)

        # Find the keypoints and descriptors with SIFT
        keypoint, descriptor = self.sift.detectAndCompute(image, None)
        return ImageFeature(filename, image, keypoint, descriptor)

    def get_image_feature(self, filename, image):
        # Find the keypoints and descriptors with SIFT
        keypoint, descriptor = self.sift.detectAndCompute(image, None)
        return ImageFeature(filename, image, keypoint, descriptor)