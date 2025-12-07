import cv2 
import numpy as np

def segment(frame):
    low_green = np.array([30, 100, 45])
    up_green = np.array([85, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask = cv2.inRange(hsv, low_green, up_green)
    green_mask1 = cv2.erode(green_mask, kernel, iterations=1)
    green_mask2 = cv2.dilate(green_mask1, kernel, iterations=6)

    contours, _ = cv2.findContours(green_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
    largest_contour = max(contours, key=cv2.contourArea)
    
    hull = cv2.convexHull(largest_contour)

    # Buat mask untuk ROI
    mask = np.zeros_like(green_mask2)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)  # Isi area dalam hull
    
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return roi_frame

low_white = np.array([0, 0, 165])
up_white = np.array([300, 60, 255])

low_green = np.array([35, 100, 100])
up_green = np.array([85, 255, 255])

cap = cv2.VideoCapture('recording3.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = segment(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred_bin = cv2.GaussianBlur(hsv, (5,5), 2)
    #bin_image = cv2.inRange(blurred_bin, low_green, up_green)
    #morph_close = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(blurred_bin, 50, 150)
    dilated_edges = cv2.dilate(edge, kernel, iterations=1)

      
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=200, minLineLength=1, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    # Display the result
    #cv2.imshow('ori', frame)
    cv2.imshow('edge', edge)
    #cv2.imshow('edge max', dilated_edges)
    #cv2.imshow('bin', blurred_bin)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()