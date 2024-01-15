import cv2
import numpy as np


def preProcessing(img):
    imgBlur = cv2.GaussianBlur(img, (9, 9), 3)  # Blur the image to remove noise
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)  # Convert to HSV
    h_min, h_max = 0, 18
    s_min, s_max = 158, 255
    v_min, v_max = 68, 255
    lower = np.array([h_min, s_min, v_min])  # Lower bound for the HSV filter
    upper = np.array([h_max, s_max, v_max])  # Upper bound for the HSV filter
    mask = cv2.inRange(imgHSV, lower, upper)  # Create a mask
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # Threshold the mask
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )  # Morphological closing
    # cv2.imshow("mask", mask)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    imgGray = cv2.bitwise_and(imgGray, imgGray, mask=mask) # Apply the mask to the grayscale image

    imgThresh = cv2.adaptiveThreshold(
        imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    ) # Adaptive thresholding
    return imgThresh 
