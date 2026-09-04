# Isolated YOLO-World test container

This directory is independent of `kuavo_ros_application`. The image contains
Ultralytics/YOLO-World only. At runtime it read-mounts the validated Jetson
PyTorch environment and CUDA libraries; nothing in the host venv or ROS
workspace is modified.

The image contains Ultralytics/YOLO-World only. ROS1, PyTorch, torchvision,
CUDA, and the camera remain supplied by read-only Jetson host mounts.

## ROS test

On the Jetson:

```bash
cd ~/yolo_world_docker
sudo docker compose up -d
sudo docker compose logs -f
```

The service subscribes to `/camera/color/image_raw` and publishes:

```text
/yolo_world/debug/image   sensor_msgs/Image
/yolo_world/detections    std_msgs/String  # JSON detections
```

The current Compose defaults are prompt `box`, `conf=0.25`, `iou=0.45`, and
`imgsz=640`. Stop the test with `sudo docker compose down`.

The standalone image tester is `infer_image.py`; it writes an annotated image
and JSON summary for a supplied still image.
