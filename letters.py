import numpy as np

import cv2
from kerasNN import Network


def find(image):
    _, threshold = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    cropped_threshold = threshold[:, 0:145]
    _, contours, _ = cv2.findContours(
        cropped_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letters_bounds = contours
    if len(letters_bounds) != 4:
        _, threshold = cv2.threshold(image, 185, 255, cv2.THRESH_BINARY)
        cropped_threshold = threshold[:, 0:145]
        _, contours, _ = cv2.findContours(
            cropped_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        letters_bounds = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (h > 15) & (w > 5) & (x > 2):
                if w / h > 1.20:
                    if w / h > 2:
                        letter_width = int(w/3)
                        letters_bounds.append((x, y, w, h))
                        letters_bounds.append(
                            (x+letter_width, y, letter_width, h))
                        letters_bounds.append(
                            (x+letter_width*2, y, letter_width, h))
                    else:
                        letter_width = int(w/2)
                        letters_bounds.append((x, y, w, h))
                        letters_bounds.append(
                            (x+letter_width, y, letter_width, h))
                else:
                    letters_bounds.append((x, y, w, h))
    return sorted(letters_bounds, key=lambda letter_bound: letter_bound[0])


def resize(image):
    return cv2.resize(image, (46, 46))
