cmake_minimum_required(VERSION 3.5.0)
project(LUOYANG VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR /opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
include_directories(LUOYANG ${ONNXRUNTIME_ROOT}/include)

include_directories(include)


add_executable(luoyang_yolo Yolo.cpp
src/Detect.cpp
src/Include.cpp
src/Model.cpp
src/Transformer.cpp)


target_link_libraries(luoyang_yolo ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so )
target_link_libraries(luoyang_yolo ${OpenCV_LIBS})

add_executable(luoyang_yolo_face YoloFace.cpp
src/Detect.cpp
src/FaceDetect.cpp
src/Include.cpp
src/Model.cpp
src/Transformer.cpp
src/FaceTransformer.cpp)

target_link_libraries(luoyang_yolo_face ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so )
target_link_libraries(luoyang_yolo_face ${OpenCV_LIBS})

add_executable(luoyang_yolo_pose YoloPose.cpp
src/Detect.cpp
src/FaceDetect.cpp
src/PoseDetect.cpp
src/Include.cpp
src/Model.cpp
src/Transformer.cpp
src/FaceTransformer.cpp
src/PoseTransformer.cpp)

target_link_libraries(luoyang_yolo_pose ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so )
target_link_libraries(luoyang_yolo_pose ${OpenCV_LIBS})


add_executable(luoyang_yolo_job YoloJob.cpp
src/Detect.cpp
src/Include.cpp
src/Model.cpp
src/Transformer.cpp)


target_link_libraries(luoyang_yolo_job ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so )
target_link_libraries(luoyang_yolo_job ${OpenCV_LIBS})
target_link_libraries(luoyang_yolo_job -lpthread)
