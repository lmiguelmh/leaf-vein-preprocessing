"""
@author lmiguelmh
@since 20170512
"""

import os
import cv2
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import hf
import filters
import contours
import contrast
from skimage import morphology, img_as_ubyte


def get_files(dir, ext):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(ext)]


def process_leaf(img_path, write_leaf=True):
    print(img_path)
    gray = cv2.imread(img_path, 0)  # gray-scale loading
    h, w = gray.shape

    # operation += "+gamma"
    gray = contrast.adjust_gamma(gray, 10.)  # todo modificar este valor a menor porque añade bastante ruido

    # operation += "+kirsch"
    gray = filters.kirsch_filter(gray)
    if write_leaf:
        leaf_path = img_path + "-1filt.jpg"
        cv2.imwrite(leaf_path, gray)
        print(leaf_path, ' file written')

    # operation += "+gamma"
    # gray = contrast.adjust_gamma(gray, 0.2)

    # operation += "+blur"
    # gray = cv2.blur(gray, (5, 5), borderType=cv2.BORDER_REPLICATE)

    # operation += "+adaptive"
    # window_size = (w >> 1) + (1 - ((w >> 1) % 2))
    window_size = (w) + (1 - (w % 2))
    # window_size = int(w/4) + (1 - (int(w/4)% 2))
    gray_bin = cv2.adaptiveThreshold(gray, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, window_size, 0)
    # gray_bin = cv2.adaptiveThreshold(gray, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_MEAN_C, window_size, 0)

    # operation += "+clean"
    # gray_bin = contours.clean_img_bin(gray_bin, 20, h * w * 0.90, 30)

    # operation += "+del"
    _, cnts, _ = cv2.findContours(gray_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    for i, cnt in enumerate(cnts):
        contour_area = cv2.contourArea(cnt)
        if contour_area < 30:
            cv2.drawContours(mask, cnts, i, (255, 255, 255), cv2.FILLED, 8)
    gray_bin = cv2.bitwise_or(gray_bin, mask)

    if write_leaf:
        leaf_path = img_path + "-2bin.jpg"
        cv2.imwrite(leaf_path, gray_bin)
        print(leaf_path, ' file written')

    # operation += "+erode"
    # gray_bin = cv2.dilate(gray_bin, cv2.getStructuringElement(cv2.MORPH_ERODE, (2, 2)), iterations=1)

    # operation += "+skel"
    gray_bin = cv2.bitwise_not(gray_bin)
    gray_bin = morphology.medial_axis(gray_bin)
    gray_bin = img_as_ubyte(gray_bin)

    if write_leaf:
        leaf_path = img_path + "-3vein.jpg"
        cv2.imwrite(leaf_path, gray_bin)
        print(leaf_path, ' file written')

    return gray_bin
