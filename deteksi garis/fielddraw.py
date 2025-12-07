import cv2
import numpy as np

def local_point(x, y, width, height, resolusi):
    center_x, center_y = width // 2, height // 2    #titik 0, 0 kanvas

    cv2.rectangle(canvas, (0 + resolusi, resolusi), (width - resolusi, height - resolusi), (255, 255, 255), 2)  #garis pinggir
    cv2.line(canvas, (0 + resolusi, center_y), (width - resolusi, center_y), (255, 255, 255), 2)    #garis tengah
    cv2.circle(canvas, (center_x, center_y), int(1.5*resolusi), (255, 255, 255), 2) #lingkaran tengah
    cv2.rectangle(canvas, (int(0.5*resolusi) + resolusi, 2*resolusi), (width - int(1.5*resolusi), resolusi), (255, 255, 255), 2) #kotak gawang 1
    #kotak bawah
    cv2.rectangle(canvas, (int(0.5*resolusi) + resolusi, height - 2*resolusi), (width - int(1.5*resolusi), height - resolusi), (255, 255, 255), 2) #kotak gawang 2

    #posisi dalam meter
    x = int(x*resolusi + center_x)
    y = int(-y*resolusi + center_y)
    cv2.circle(canvas, (x, y), int(0.2 * resolusi), (0, 0, 255), -1)

#create kanvas
resolusi = 50
width, height = 8, 11
width *= resolusi
height *= resolusi
canvas = np.zeros((height, width, 3), dtype=np.uint8)
local_point(-1, -4.5, width, height, resolusi)
cv2.imshow('Canvas', canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()
