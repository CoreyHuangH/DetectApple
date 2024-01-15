import cv2
import numpy as np


def findContours(imgTresh, original_img):
    contours, hierarchy = cv2.findContours(
        imgTresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    ) # Find the contours
    imgContours = original_img.copy() # Copy the original image
    for cnt in contours:
        area = cv2.contourArea(cnt) # Calculate the area of the contour
        if area > 1200:
            # cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3) # Draw the contours
            peri = cv2.arcLength(cnt, True) # Calculate the perimeter of the contour
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Approximate the shape of the contour
            x, y, w, h = cv2.boundingRect(approx) # Get the bounding rectangle of the contour
            cv2.rectangle(imgContours, (x, y - 20), (x + w, y + h + 10), (0, 255, 0), 2) # Draw the bounding rectangle
            cv2.putText(
                imgContours,
                "Apple",
                (x + w // 2 - 45, y + h // 2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.5,
                (0, 255, 0),
                1,
            ) # Put the text "Apple" on the image
    return imgContours
