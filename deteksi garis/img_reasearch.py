import cv2
import numpy as np

def interpolate_line(x1, y1, x2, y2):
    line_length = int(np.hypot(x2 - x1, y2 - y1))
    x_coords = np.linspace(x1, x2, line_length).astype(int)
    y_coords = np.linspace(y1, y2, line_length).astype(int)
    return list(zip(x_coords, y_coords))


# Contoh penggunaan
x1, y1, x2, y2 = 10, 10, 20, 30
points = interpolate_line(x1, y1, x2, y2)
print("Titik-titik pada garis:", points)


# Gambar kosong
frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

# Koordinat garis
x1, y1, x2, y2 = 100, 100, 800, 600
cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Hitung titik-titik garis
points = interpolate_line(x1, y1, x2, y2)  # atau bresenham_line(x1, y1, x2, y2)

# Gambarkan titik-titik pada frame
for x, y in points:
    frame[y, x] = (0, 255, 0)  # Warna hijau

# Tampilkan hasil
cv2.imshow("Garis", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
