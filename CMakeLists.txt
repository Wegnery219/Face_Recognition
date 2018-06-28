cmake_minimum_required(VERSION 2.8.12)
project(Test4)
add_subdirectory(E:/shixun/face_recognition/dlib-19.13/dlib dlib_build)
set(OpenCV_DIR D:/opencv3.1.0/opencv/build)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Test4 Test4.cpp)
target_link_libraries(Test4 dlib::dlib)
target_link_libraries(Test4 ${OpenCV_LIBS})



