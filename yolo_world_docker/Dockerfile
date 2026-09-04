FROM nvcr.io/nvidia/l4t-base:35.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/opt/mplconfig \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# Keep the image small. PyTorch/torchvision and the scientific Python stack
# are supplied read-only from the already validated Jetson venv at runtime.
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

# Ultralytics 8.2.2 contains the YOLO-World API while remaining compatible
# with the Python 3.8 / PyTorch 2.0 stack already installed on the Jetson.
# --no-deps is intentional: dependencies, including Jetson's torch build,
# come from the read-only host site-packages mount when the container runs.
RUN python3 -m pip install --no-cache-dir --no-deps \
        --target /opt/yolo_world_site \
        ultralytics==8.2.2 \
    && python3 -m pip install --no-cache-dir --no-deps \
        --target /opt/yolo_world_site \
        https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip \
    && python3 -m pip install --no-cache-dir --no-deps \
        --target /opt/yolo_world_site \
        ftfy==6.1.1 regex==2023.8.8 wcwidth==0.2.13 \
    && mkdir -p /opt/mplconfig /tmp/ultralytics \
    && touch /opt/mplconfig/matplotlibrc

COPY infer_image.py /opt/yolo_world/infer_image.py

WORKDIR /opt/yolo_world
ENTRYPOINT ["python3", "/opt/yolo_world/infer_image.py"]
