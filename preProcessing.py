import cv2
import numpy as np


def preProcessing(img: np.ndarray) -> np.ndarray:
    imgBlur = cv2.GaussianBlur(img, (9, 9), 3)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
    h_min, h_max = 0, 18
    s_min, s_max = 158, 255
    v_min, v_max = 68, 255
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.bitwise_and(imgGray, imgGray, mask=mask)

    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return imgThresh
