FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 设置阿里云Ubuntu镜像源
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

# 设置非交互式环境避免tzdata安装问题
ENV DEBIAN_FRONTEND=noninteractive

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    tar \
    g++ \
    make \
    pkg-config \
    libgtk-4-dev \
    libeigen3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenexr-dev \
    python3-dev \
    python3-numpy \
    libgtk2.0-dev \
    gdb \
    libboost-graph-dev \
    && rm -rf /var/lib/apt/lists/*

COPY opencv-4.8.0.tar.gz /tmp/
COPY onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz /tmp/
COPY eigen-3.4.0.tar.gz /tmp/

# 安装Eigen3
RUN tar -xzf /tmp/eigen-3.4.0.tar.gz -C /tmp/ && \
    mkdir -p /tmp/eigen-3.4.0/build && \
    cd /tmp/eigen-3.4.0/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make install && \
    ldconfig && \
    rm -rf /tmp/eigen-3.4.0 /tmp/eigen-3.4.0.tar.gz

# 安装ONNX Runtime
RUN tar -xzf /tmp/onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz -C /usr/local/ && \
    mv /usr/local/onnxruntime-linux-x64-gpu-1.18.1 /usr/local/onnxruntime && \
    echo "/usr/local/onnxruntime/lib" >> /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig && \
    rm /tmp/onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz

# 编译安装OpenCV
RUN tar -xzf /tmp/opencv-4.8.0.tar.gz -C /tmp/ && \
    mv /tmp/opencv-4.8.0 /opencv && \
    mkdir -p /opencv/build && \
    cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=YES .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /opencv /tmp/opencv-4.8.0.tar.gz