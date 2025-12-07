import cv2
import numpy as np

def field(frame):
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

    return mask

cap = cv2.VideoCapture('sample4.mkv')


if not cap.isOpened():
    print("Error: No Video Opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask_field = field(frame)

    cv2.imshow('masking', mask_field)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()