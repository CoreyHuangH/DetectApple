import cv2
import numpy as np


def findContours(imgTresh: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(
        imgTresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    imgContours = original_img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1200:
            # cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContours, (x, y - 20), (x + w, y + h + 10), (0, 255, 0), 2)
            cv2.putText(
                imgContours,
                "Apple",
                (x + w // 2 - 45, y + h // 2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.5,
                (0, 255, 0),
                1,
            )
    return imgContours
