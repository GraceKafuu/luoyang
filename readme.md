## 模型文件
模型文件均来自于 https://github.com/zhangluoyang/Yolo
预训练权重已经训练好的模型文件等可以通过网盘下载
链接: https://pan.baidu.com/s/1a65cgfZ0FHX6RUw4yEJnng?pwd=1234 提取码: 1234
## 下载代码
cd ~
git clone https://github.com/zhangluoyang/Yolo.git

## 制作镜像
将下载好的 onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz 和 opencv-4.8.0.tar.gz 放在 docker 目录下
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
./luoyang_yolo /home/zhangluoyang/yolo_model/yolo_v6 /home/zhangluoyang/person.png

./luoyang_yolo_face /home/zhangluoyang/yolo_model/yolo_v6_face /home/zhangluoyang/person.png

./luoyang_yolo_pose /home/zhangluoyang/yolo_model/yolo_v6_pose /home/zhangluoyang/person.png

./luoyang_yolo_job /home/zhangluoyang/yolo_model/yolo_v6 1 /home/zhangluoyang/caiji.mp4
