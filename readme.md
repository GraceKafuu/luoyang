## 下载代码
cd ~
git clone 

## 制作镜像
cd docker
docker build -t luoyang:v0.0.1 .

## 挂载宿主机的 X11 显示
xhost +local:root  
## 启动docer
docker run -itd --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home:/home --name luoyang -p 11945:11945 luoyang:v0.0.1

docker exec -it luoyang bash

## 编译
cd /home/zhangluoyang/luoyang
mkdir build
cd build
cmake ..
make -j2
## 运行
### 目标检测
./luoyang_yolo /home/zhangluoyang/yolo_model/yolo_v10 /home/zhangluoyang/sheet.jpeg
### 人脸检测
./luoyang_yolo_face /home/zhangluoyang/yolo_model/yolo_v6_face /home/zhangluoyang/person.png
### 姿态估计
./luoyang_yolo_pose /home/zhangluoyang/yolo_model/yolo_v6_pose /home/zhangluoyang/person.png
### 目标跟踪
./luoyang_yolo_track /home/zhangluoyang/yolo_model/yolo_v8 /home/zhangluoyang/workspace/luoyang/resource/palace.mp4


