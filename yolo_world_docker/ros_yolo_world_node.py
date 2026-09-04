#!/usr/bin/env python3
"""ROS1 subscriber/publisher wrapper for a YOLO-World prompt detector."""

import json
import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloWorldRosNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_image", "/camera/color/image_raw")
        self.model_path = rospy.get_param("~model_path", "/models/yolov8s-worldv2.pt")
        self.prompt = rospy.get_param("~prompt", "box")
        self.conf = float(rospy.get_param("~conf", 0.25))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.imgsz = int(rospy.get_param("~imgsz", 640))

        self.bridge = CvBridge()
        self.frame_count = 0
        self.processed_count = 0
        self.busy_lock = threading.Lock()

        rospy.loginfo("Loading YOLO-World model: %s", self.model_path)
        self.model = YOLO(self.model_path)
        self.model.set_classes([self.prompt])
        rospy.loginfo(
            "YOLO-World ready: prompt=%s conf=%.3f iou=%.3f imgsz=%d names=%s",
            self.prompt,
            self.conf,
            self.iou,
            self.imgsz,
            self.model.names,
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
        rospy.loginfo("Publishing %s and %s", "~debug/image", "~detections")

    def image_callback(self, msg):
        # The Jetson inference is slower than the camera stream. Drop frames
        # while one inference is running so the node always works on a recent
        # image instead of building an old-message backlog.
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
                            "class_name": str(self.model.names[class_id]),
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
                "prompt": self.prompt,
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
                "YOLO-World processed=%d detections=%d elapsed=%.1f ms",
                self.processed_count,
                len(detections),
                elapsed_ms,
            )
        except Exception as exc:
            rospy.logerr_throttle(5.0, "YOLO-World callback failed: %s", exc)
        finally:
            self.busy_lock.release()


def main():
    rospy.init_node("yolo_world", anonymous=False)
    YoloWorldRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
