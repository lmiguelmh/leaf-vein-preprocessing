"""
@author lmiguelmh
@since 20170514
"""

import cv2
import numpy as np


def clean_img_bin(img_bin, min_area, max_area, border_radius=10):
    h, w = img_bin.shape

    # add a "border" so a char thats in the borders is not detected like a outer contour
    img_bin[0, :] = 255
    img_bin[h - 1, :] = 255
    img_bin[:, 0] = 255
    img_bin[:, w - 1] = 255

    _, character_contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    for i, character_cnt in enumerate(character_contours):
        rx, ry, rw, rh = cv2.boundingRect(character_cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1
        contour_area = cv2.contourArea(character_cnt)
        has_right_area = min_area < contour_area < max_area
        close_to_border = rxf < border_radius or ryf < border_radius or rx + border_radius > w or ry + border_radius > h
        if has_right_area and not close_to_border:
            cv2.drawContours(mask, character_contours, i, (255, 255, 255), cv2.FILLED, 8)

    final2 = cv2.bitwise_and(img_bin, img_bin, mask=mask)
    bk = cv2.bitwise_not(np.zeros((h, w), np.uint8))
    final2_bk = cv2.bitwise_and(bk, bk, mask=cv2.bitwise_not(mask))
    return cv2.bitwise_or(final2, final2_bk)
