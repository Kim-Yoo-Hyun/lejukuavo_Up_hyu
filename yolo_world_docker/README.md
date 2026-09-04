# Isolated YOLO-World test container

This directory is independent of `kuavo_ros_application`. The image contains
Ultralytics/YOLO-World only. At runtime it read-mounts the validated Jetson
PyTorch environment and CUDA libraries; nothing in the host venv or ROS
workspace is modified.

The first test uses a host-side ROS snapshot (`/camera/color/image_raw`) so the
container does not need to install or alter ROS packages. The container runs
YOLO-World with the prompt `cardboard box` and writes an annotated image plus a
JSON detection summary.
