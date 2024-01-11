import cv2
import numpy as np
from preProcessing import preProcessing
from findContours import findContours


def main():
    img = cv2.imread("apple.png")
    # cv2.imshow("Original", img)
    imgTresh = preProcessing(img) # Preprocess the image
    # cv2.imshow("Preprocessed", imgTresh)
    imgContours = findContours(imgTresh, img) # Find the contours
    cv2.imshow("Contours", imgContours)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
