cmake_minimum_required(VERSION 3.5.0)
project(LUOYANG VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)
# 查找Boost库
find_package(Boost REQUIRED)

set(OpenCV_DIR /opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
include_directories(LUOYANG ${ONNXRUNTIME_ROOT}/include)

# 添加Boost包含目录
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)
# 添加Eigen3包含目录
include_directories(${EIGEN3_INCLUDE_DIR})


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
src/Transformer.cpp
src/BytekalmanFilter.cpp)


target_link_libraries(luoyang_yolo_job ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so )
target_link_libraries(luoyang_yolo_job ${OpenCV_LIBS})
target_link_libraries(luoyang_yolo_job -lpthread)

# 添加ByteTrack目标跟踪器
add_executable(luoyang_yolo_track YoloTrack.cpp
src/Detect.cpp
src/Include.cpp
src/Model.cpp
src/Transformer.cpp
src/BYTETracker.cpp
src/BytekalmanFilter.cpp
src/STrack.cpp
src/lapjv.cpp)

target_compile_definitions(luoyang_yolo_track PRIVATE DEBUG)
target_link_libraries(luoyang_yolo_track ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so )
target_link_libraries(luoyang_yolo_track ${OpenCV_LIBS})
# 链接Boost库
target_link_libraries(luoyang_yolo_track ${Boost_LIBRARIES})
target_link_libraries(luoyang_yolo_track Eigen3::Eigen)