import cv2
import numpy as np

# Membaca gambar
frame = cv2.imread('line1.png')  # Ganti dengan path gambar yang sesuai

# Mendapatkan dimensi frame (lebar dan tinggi)
height, width = frame.shape[:2]

# Tentukan titik tengah frame (horizontal)
center_x = width // 2
center_y = height  # Titik tengah bawah adalah bagian bawah dari gambar

# Tentukan jarak x
x = 200  # Jarak antar garis vertikal (dapat diubah sesuai keinginan)

# Posisi garis vertikal
line1_x = center_x  # Garis pertama di tengah
line2_x = center_x + x  # Garis kedua di kanan dengan jarak x
line3_x = center_x - 2 * x  # Garis ketiga di kiri dengan jarak 2x

# Gambar garis vertikal
cv2.line(frame, (line1_x, 0), (line1_x, height), (0, 0, 255), 2)  # Garis tengah
cv2.line(frame, (line2_x, 0), (line2_x, height), (0, 255, 0), 2)  # Garis kanan
cv2.line(frame, (line3_x, 0), (line3_x, height), (255, 0, 0), 2)  # Garis kiri

# Menggambar garis miring ke kiri dengan sudut 66 derajat dari titik tengah bawah
angle_left = 13  # Sudut 66 derajat ke kiri
length = height  # Panjang garis yang tidak terbatas, sampai batas bawah frame

# Menghitung koordinat titik akhir garis miring kiri dengan menggunakan trigonometri
radian_left = np.deg2rad(angle_left)  # Konversi sudut ke radian
dx_left = int(length * np.cos(radian_left))  # Perubahan pada sumbu X
dy_left = int(length * np.sin(radian_left))  # Perubahan pada sumbu Y

# Titik awal garis miring kiri adalah titik tengah bawah
start_point_left = (center_x, height)

# Titik akhir garis miring kiri dihitung berdasarkan sudut dan panjang garis
end_point_left = (start_point_left[0] - dx_left, start_point_left[1] - dy_left)  # Miring ke kiri

# Gambar garis miring ke kiri
cv2.line(frame, start_point_left, end_point_left, (255, 255, 0), 2)  # Garis miring ke kiri (kuning)

# Menggambar garis miring ke kanan dengan sudut 77 derajat dari titik tengah bawah
angle_right = 13  # Sudut 77 derajat ke kanan

# Menghitung koordinat titik akhir garis miring kanan dengan menggunakan trigonometri
radian_right = np.deg2rad(angle_right)  # Konversi sudut ke radian
dx_right = int(length * np.cos(radian_right))  # Perubahan pada sumbu X
dy_right = int(length * np.sin(radian_right))  # Perubahan pada sumbu Y

# Titik awal garis miring kanan adalah titik tengah bawah
start_point_right = (center_x, height)

# Titik akhir garis miring kanan dihitung berdasarkan sudut dan panjang garis
end_point_right = (start_point_right[0] + dx_right, start_point_right[1] - dy_right)  # Miring ke kanan

# Gambar garis miring ke kanan
cv2.line(frame, start_point_right, end_point_right, (0, 255, 255), 2)  # Garis miring ke kanan (cyan)

# Menampilkan gambar dengan tiga garis vertikal dan dua garis miring
cv2.imshow('Frame with Vertical and Slanted Lines', frame)

# Tunggu hingga tombol ditekan untuk menutup jendela
cv2.waitKey(0)
cv2.destroyAllWindows()
