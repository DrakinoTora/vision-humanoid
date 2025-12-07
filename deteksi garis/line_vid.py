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

    mask = np.zeros_like(green_mask2)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return roi_frame

def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1 + 1e-6)

def is_parallel_and_close(line1, line2, slope_threshold=0.1, distance_threshold=10):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    slope1 = calculate_slope(x1, y1, x2, y2)
    slope2 = calculate_slope(x3, y3, x4, y4)

    if abs(slope1 - slope2) > slope_threshold:
        return False

    dist = abs((y3 - slope1 * x3 - (y1 - slope1 * x1)) / np.sqrt(1 + slope1**2))
    return dist < distance_threshold

def merge_lines(lines):
    merged_lines = []

    while lines:
        line = lines.pop(0)
        x1, y1, x2, y2 = line[0]
        max_line = line
        max_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        to_remove = []
        for other_line in lines:
            x3, y3, x4, y4 = other_line[0]
            if is_parallel_and_close((x1, y1, x2, y2), (x3, y3, x4, y4)):
                length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                if length > max_length:
                    max_length = length
                    max_line = other_line
                to_remove.append(other_line)

        for line_to_remove in to_remove:
            lines.remove(line_to_remove)

        merged_lines.append(max_line)
    return merged_lines

vid = cv2.VideoCapture('sample3.mp4')
while True:
    ret, frame = vid.read()
    if not ret:
        break
    
    frame = segment(frame)

    low_white = np.array([0, 0, 165])
    up_white = np.array([300, 60, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, low_white, up_white)
    blurred_bin = cv2.GaussianBlur(hsv, (5, 5), 2)
    edges = cv2.Canny(blurred_bin, 50, 150)

    lines = cv2.HoughLinesP(mask_white, 1, np.pi/180, threshold=10, minLineLength=50, maxLineGap=10)

    if lines is not None:
        merged_lines = merge_lines(lines.tolist())

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Detected Lines', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()