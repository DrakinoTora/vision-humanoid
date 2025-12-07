import cv2 as cv
import numpy as np

imgframe = cv.imread('sample4.png')

if imgframe is None:
    print("Error: Can't load image")
else:
    hsv = cv.cvtColor(imgframe, cv.COLOR_BGR2HSV)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    valid_mask = cv.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))

    hist = cv.calcHist([hsv], [0], valid_mask, [180], [0, 180])

    dominant_hue = np.argmax(hist)
    
    low_green = np.array([dominant_hue - 30, 100, 100])
    up_green = np.array([dominant_hue + 30, 255, 255])

    green_mask = cv.inRange(hsv, low_green, up_green)

    low_white = np.array([0, 0, 200])
    up_white = np.array([180, 30, 255])
    white_mask = cv.inRange(hsv, low_white, up_white)

    combined_mask = cv.bitwise_or(green_mask, white_mask)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        hull = cv.convexHull(largest_contour)

        mask = np.zeros_like(combined_mask)
        cv.drawContours(mask, [hull], -1, 255, thickness=-1)

        boundary_frame = imgframe.copy()
        cv.drawContours(boundary_frame, [hull], -1, (0, 255, 0), thickness=2)

        roi_frame = cv.bitwise_and(imgframe, imgframe, mask=mask)

        cv.imshow('Field Only', boundary_frame)
        cv.imshow('ROI Frame', roi_frame)
        cv.imshow('Filled Mask', combined_mask)

        cv.waitKey(0)

cv.destroyAllWindows()