cmake_minimum_required(VERSION 3.10)
project(BallDetection)

find_package(OpenCV REQUIRED)

add_executable(BallDetection ball_final.cpp)
target_link_libraries(BallDetection ${OpenCV_LIBS})
