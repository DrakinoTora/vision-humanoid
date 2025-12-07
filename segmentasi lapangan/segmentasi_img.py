import cv2 as cv
import numpy as np

imgframe = cv.imread('sample4.png')

if imgframe is None:
    print("Error: Can't load image")
else:
    low_green = np.array([35, 100, 100])
    up_green = np.array([85, 255, 255])

    hsv = cv.cvtColor(imgframe, cv.COLOR_BGR2HSV)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    green_mask = cv.inRange(hsv, low_green, up_green)
    
    green_mask1 = cv.erode(green_mask, kernel, iterations=1)
    green_mask2 = cv.dilate(green_mask1, kernel, iterations=6)

    contours, _ = cv.findContours(green_mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)

        hull = cv.convexHull(largest_contour)

        mask = np.zeros_like(green_mask2)
        cv.drawContours(mask, [hull], -1, 255, thickness=-1)

        boundary_frame = imgframe.copy()
        cv.drawContours(boundary_frame, [hull], -1, (0, 0, 255), thickness=2)

        roi_frame = cv.bitwise_and(imgframe, imgframe, mask=mask)

        #cv.imshow('Field Only', boundary_frame)
        cv.imshow('mask1', roi_frame)
        #cv.imshow('mask', green_mask2)
        
        cv.waitKey(0)

cv.destroyAllWindows()