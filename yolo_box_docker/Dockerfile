FROM nvcr.io/nvidia/l4t-base:35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/opt/mplconfig \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# PyTorch, torchvision, CUDA libraries, and ROS1 are supplied through
# read-only Jetson host mounts at runtime. This keeps this best.pt test
# isolated from the existing host environment and from YOLO-World.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3-pip \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# best.pt was inspected with Ultralytics 8.0.200 and is a two-class detect
# model ({0: label, 1: package}). Install that exact Python package version
# inside this image without replacing the Jetson torch build.
RUN python3 -m pip install --no-cache-dir --no-deps \
        --target /opt/yolo_box_site \
        ultralytics==8.0.200 \
    && mkdir -p /opt/mplconfig /tmp/ultralytics \
    && touch /opt/mplconfig/matplotlibrc

COPY best_pt_infer.py /opt/yolo_box/best_pt_infer.py
COPY ros_best_pt_node.py /opt/yolo_box/ros_best_pt_node.py
COPY models/best.pt /models/best.pt
COPY original_package /opt/yolo_box/original_package

WORKDIR /opt/yolo_box
ENTRYPOINT ["python3", "/opt/yolo_box/ros_best_pt_node.py"]
