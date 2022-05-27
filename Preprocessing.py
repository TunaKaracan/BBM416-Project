import numpy as np
import cv2


# TODO segmentation and image augmentation
def segmentate_image_kmeans(img):
    two_d_img = img.reshape((-1, 1))
    two_d_img = np.float32(two_d_img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k, attempts = 2, 20
    ret, label, center = cv2.kmeans(two_d_img, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_img = res.reshape(img.shape)
    return result_img


def segmentate_image(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, 200, 200)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img


