#!/usr/bin/env python3
"""ROS1 subscriber/publisher wrapper for the existing best.pt detector."""

import json
import threading
import time

import cv2  # Load OpenCV before cv_bridge on this Jetson image.
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


def class_id_for(names, target):
    target = str(target)
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in items:
        if str(class_name) == target:
            return int(class_id)
    return None


class BestPtRosNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_image", "/camera/color/image_raw")
        self.model_path = rospy.get_param("~model_path", "/models/best.pt")
        self.target_class = rospy.get_param("~target_class", "package")
        self.conf = float(rospy.get_param("~conf", 0.25))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.imgsz = int(rospy.get_param("~imgsz", 640))

        self.bridge = CvBridge()
        self.busy_lock = threading.Lock()
        self.processed_count = 0

        rospy.loginfo("Loading best.pt model: %s", self.model_path)
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        self.target_id = class_id_for(self.class_names, self.target_class)
        if self.target_id is None:
            raise RuntimeError(
                "target class '%s' not found in model names %s"
                % (self.target_class, self.class_names)
            )
        rospy.loginfo(
            "best.pt ready: task=%s names=%s target=%s(id=%d) conf=%.3f iou=%.3f imgsz=%d",
            self.model.task,
            self.class_names,
            self.target_class,
            self.target_id,
            self.conf,
            self.iou,
            self.imgsz,
        )

        self.debug_pub = rospy.Publisher("~debug/image", Image, queue_size=1)
        self.detections_pub = rospy.Publisher("~detections", String, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.loginfo("Subscribed to %s", self.input_topic)
        rospy.loginfo("Publishing ~debug/image and ~detections")

    def image_callback(self, msg):
        # Jetson inference and camera input run at different rates. Drop a
        # frame when an inference is active rather than accumulating stale
        # images in the callback queue.
        if not self.busy_lock.acquire(False):
            return

        try:
            started = time.perf_counter()
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            result = self.model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=0,
                classes=[self.target_id],
                verbose=False,
            )[0]

            annotated = result.plot(conf=True, labels=True, boxes=True)
            debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": str(self.class_names[class_id]),
                            "confidence": round(float(box.conf[0].item()), 4),
                            "xyxy": [
                                round(float(value), 1)
                                for value in box.xyxy[0].tolist()
                            ],
                        }
                    )

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            payload = {
                "stamp": msg.header.stamp.to_sec(),
                "frame_id": msg.header.frame_id,
                "input_topic": self.input_topic,
                "model": self.model_path,
                "target_class": self.target_class,
                "conf_threshold": self.conf,
                "iou_threshold": self.iou,
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "elapsed_ms": round(elapsed_ms, 1),
                "detections": detections,
            }
            self.detections_pub.publish(String(data=json.dumps(payload)))
            self.processed_count += 1
            rospy.loginfo_throttle(
                10.0,
                "best.pt processed=%d detections=%d elapsed=%.1f ms",
                self.processed_count,
                len(detections),
                elapsed_ms,
            )
        except Exception as exc:
            rospy.logerr_throttle(5.0, "best.pt callback failed: %s", exc)
        finally:
            self.busy_lock.release()


def main():
    rospy.init_node("yolo_box_best", anonymous=False)
    BestPtRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
