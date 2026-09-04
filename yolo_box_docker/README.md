# Isolated best.pt Docker test

This is separate from `yolo_world_docker`. It copies the existing
`yolo_box_object_detection` model into `models/best.pt` and runs it with the
validated Jetson PyTorch/CUDA environment mounted read-only at runtime.

The inspected model is an Ultralytics detection model with:

```text
{0: label, 1: package}
```

The Compose service targets class `package`, matching the original ROS code.

## Run

```bash
cd ~/yolo_box_docker
sudo docker compose up -d
sudo docker compose logs -f
```

Input:

```text
/camera/color/image_raw
```

Outputs:

```text
/yolo_box_best/debug/image   sensor_msgs/Image
/yolo_box_best/detections    std_msgs/String  # JSON
```

Stop it with:

```bash
sudo docker compose down
```
