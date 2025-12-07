import cv2
import numpy as np

def ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([1, 80, 20])
    upper_orange = np.array([25, 255, 255])

    mask_ball = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((15, 15), np.uint8)
    mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, kernel)

    return mask_ball

def field(frame):
    
    low_green = np.array([35, 100, 100])
    up_green = np.array([85, 255, 255])

    if frame is None:
        raise ValueError("Invalid frame")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, low_green, up_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask1 = cv.erode(green_mask, kernel, iterations=1)
    # green_mask_cleaned = cv.dilate(mask1, kernel, iterations=6)
    green_mask_cleaned = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations= 3)

    contours, _ = cv2.findContours(green_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    roi_mask = np.zeros(green_mask_cleaned.shape, dtype=np.uint8)
    if largest_contour is not None:
        cv2.drawContours(roi_mask, [cv2.convexHull(largest_contour)], -1, 255, thickness=-1)
    
    if largest_contour is None:
        return frame, green_mask_cleaned

    roi_image = cv2.bitwise_and(frame, frame, mask=roi_mask)

    return roi_image, green_mask_cleaned

def find_first_last_orange(line_data, line_start):
    first_orange = None
    last_orange = None
    
    for idx in range(line_data.shape[0]): 
        if line_data[idx] == 255:
            first_orange = line_start + idx 
            break
    
    for idx in range(line_data.shape[0] - 1, -1, -1):  
        if line_data[idx] == 255:
            last_orange = line_start + idx 
            break
    
    return first_orange, last_orange

def find_top_bottom_orange(column_data, col_start):
    top_orange = None
    bot_orange = None
    
    for idy in range(column_data.shape[0]): 
        if column_data[idy] == 255:
            top_orange = col_start + idy 
            break
    
    for idy in range(column_data.shape[0] - 1, -1, -1):  
        if column_data[idy] == 255:
            bot_orange = col_start + idy 
            break
    
    return top_orange, bot_orange

def detect(mask_ball, mask_field, frame):
    blurred_ball_mask = cv2.GaussianBlur(mask_ball, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred_ball_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=1000
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            stroke = int(1.1 * r)
            
            # Horizontal line processing
            line_y = y
            line_x_start = max(0, x - stroke)
            line_x_end = min(mask_ball.shape[1] - 1, x + stroke)
            orange_hline = mask_ball[line_y, line_x_start:line_x_end]
            first_orange, last_orange = find_first_last_orange(orange_hline, line_x_start)
            
            if first_orange is None or last_orange is None:
                continue
            
            total_x_pixel = last_orange - first_orange
            r_new = int(total_x_pixel / 2)
            x_new = first_orange + r_new
            
            # Vertical line processing
            line_x = x_new
            line_y_start = int(y) - int(stroke)
            line_y_end = int(y) + int(stroke)
            orange_vline = mask_ball[line_y_start:line_y_end, line_x]
            top_orange, bot_orange = find_top_bottom_orange(orange_vline, line_y_start)
            
            if top_orange is None or bot_orange is None:
                continue
            
            total_y_pixel = abs(top_orange - bot_orange)
            y_new = bot_orange - int(total_y_pixel / 2)
            
            if y_new != y:
                line_y = y_new
                orange_hline = mask_ball[line_y, line_x_start:line_x_end]
                first_orange, last_orange = find_first_last_orange(orange_hline, line_x_start)
                
                if first_orange is None or last_orange is None:
                    continue
                
                total_x_pixel = last_orange - first_orange
                r_new = int(total_x_pixel / 2)
                x_new = first_orange + r_new

            R = int(r_new * 1.5)  #jarak deteksi
            
            x1, y1 = max(x_new - R, 0), max(y_new - R, 0)
            x2, y2 = min(x_new + R, frame.shape[1]), min(y_new + R, frame.shape[0])
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            surrounding_field = mask_field[y1:y2, x1:x2]
            field_ratio = np.sum(surrounding_field == 255) / surrounding_field.size

            surrounding_ball = mask_ball[y1:y2, x1:x2]
            ball_ratio = np.sum(surrounding_ball == 255) / surrounding_ball.size

            if (field_ratio > 0.16 and ball_ratio < 0.47):
                cv2.line(frame, (x_new, y_new + r_new), (x_new, y_new - r_new), (0, 255, 0), 2)
                cv2.line(frame, (x_new - r_new, y_new), (x_new + r_new, y_new), (0, 255, 0), 2)
                cv2.circle(frame, (x_new, y_new), r_new, (0, 255, 0), 2)
            else:
                continue

            actual_diameter= 0.13 #meter
            focal_length= 714.1
            detected_diameter= total_x_pixel
            if detected_diameter==0:
                distance=0
            else:
                distance= (actual_diameter*focal_length)/detected_diameter
            text2 = f"Distance: {distance:.2f} meter"
            cv2.putText(frame, text2, (x_new - r_new, y_new + r_new + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            #print("distance= %.3f cm" %(distance*100))
            cv2.line(frame, (center_x, center_y), (x_new, y_new), (255, 255, 255), 2)

            break

    return frame

#cap = cv2.VideoCapture('sample3.mp4')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No Video Opened")
    exit()
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fps = cap.get(cv2.CAP_PROP_FPS)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#out = cv2.VideoWriter('test2.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    y_start = 0
    y_end = height
    x_start = 0
    x_end = width

    cv2.line(frame, (center_x, y_start), (center_x, y_end), (255, 255, 255), 1)
    cv2.line(frame, (x_start, center_y), (x_end, center_y), (255, 255, 255), 1)

    seg_field, mask_field = field(frame)
    mask_ball = ball(seg_field)
    #mask_ball = ball(frame)
    #final_frame = detect(mask_ball, frame, frame)
    final_frame = detect(mask_ball, mask_field, frame)

    cv2.imshow('mask field', mask_field)
    #cv2.imshow('mask ball', mask_ball)

    if final_frame is not None:
        #out.write(final_frame)
        cv2.imshow('ball detect', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#out.release()
cv2.destroyAllWindows()
