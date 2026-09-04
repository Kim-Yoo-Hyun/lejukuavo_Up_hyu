#!/usr/bin/env python3
"""Run a no-training YOLO-World prompt test on one RGB image."""

import argparse
import json
import os
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input RGB image")
    parser.add_argument("--output", required=True, help="Annotated output image")
    parser.add_argument("--model", default="/models/yolov8s-worldv2.pt")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt class; repeat this option for multiple classes",
    )
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = args.prompts or ["cardboard box"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("cannot read input image: %s" % args.image)

    started = time.time()
    model = YOLO(args.model)
    model.set_classes(prompts)
    result = model.predict(
        image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=0,
        verbose=False,
    )[0]
    elapsed_ms = (time.time() - started) * 1000.0

    annotated = result.plot(conf=True, labels=True, boxes=True)
    if not cv2.imwrite(args.output, annotated):
        raise SystemExit("cannot write output image: %s" % args.output)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": str(model.names[cls_id]),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                }
            )

    print(
        json.dumps(
            {
                "image": args.image,
                "output": args.output,
                "model": args.model,
                "prompts": prompts,
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "elapsed_ms": round(elapsed_ms, 1),
                "detections": detections,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
