import cv2
import numpy as np

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, lambda x: x)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, lambda x: x)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, lambda x: x)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, lambda x: x)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, lambda x: x)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, lambda x: x)

cv2.setTrackbarPos("Hue Min", "TrackBars", 0)
cv2.setTrackbarPos("Hue Max", "TrackBars", 18)
cv2.setTrackbarPos("Sat Min", "TrackBars", 158)
cv2.setTrackbarPos("Sat Max", "TrackBars", 255)
cv2.setTrackbarPos("Val Min", "TrackBars", 68)
cv2.setTrackbarPos("Val Max", "TrackBars", 255)

while True:
    img = cv2.imread("apple.png")
    imgBlur = cv2.GaussianBlur(img, (9, 9), 3)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
