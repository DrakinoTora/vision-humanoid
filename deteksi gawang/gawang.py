import cv2
import numpy as np

def detect_goalposts(frame):
    """
    Detects only goalposts in a given frame using HSV color segmentation, 
    contour filtering, and Hough Line Transform to remove field lines.
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # *Updated HSV Range for White Goalposts (Filters Out Field Lines)*
    lower_white = np.array([0, 0, 220])  # Increased brightness threshold
    upper_white = np.array([180, 40, 255])  # Excludes dull white (grayish lines)

    # Create binary mask where white pixels are white, everything else is black
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Edge detection to find contours of the goalpost
    edges = cv2.Canny(mask, 50, 150)

    # *Apply Hough Line Transform to detect and remove field lines*
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):  # Horizontal or diagonal lines
                cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness=10)  # Remove them

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # *Filtering based on aspect ratio, size, and verticality*
        aspect_ratio = h / float(w)  # Goalposts are tall, field lines are long & thin
        area = w * h  # Calculate area

        # *Condition to filter only goalposts:*
        if aspect_ratio > 3 and h > 70 and area > 1000:  # Stricter filtering
            filtered_contours.append(cnt)  # Keep only goalposts

    return mask, filtered_contours

def main(video_path):
    """
    Reads an MP4 video file, detects only goalposts in each frame,
    and overlays a yellow outline.
    Press 'q' to exit the program.
    """
    cap = cv2.VideoCapture(video_path)  # Load video file

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    print("Processing video... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break  # Exit loop if no more frames

        # Detect goalposts and get the mask & filtered contours
        mask, contours = detect_goalposts(frame)

        # Create an overlay for the yellow outline
        overlay = frame.copy()

        # Draw thick yellow contours around detected goalposts
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), thickness=10)  # Yellow outline

        # Blend overlay with original frame for a smooth effect
        alpha = 0.6  # Adjust transparency
        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Resize output window
        small_blended = cv2.resize(blended, (640, 360))  # Resize to 640x360
        small_mask = cv2.resize(mask, (640, 360))  # Resize the mask too

        # Display results
        cv2.imshow("Goalpost Detection - Yellow Outline", small_blended)
        cv2.imshow("Goalpost Mask", small_mask)  # Shows the binary mask for debugging

        # Exit when 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjusted delay for smooth playback
            print("Exiting program...")
            break

    # Properly release the video and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete. Windows closed.")

if __name__ == "_main_":
    video_path = r"D:\GMRT\Altair\vision\program deteksi garis\sample4.mkv"  # Use correct path format
    main(video_path)