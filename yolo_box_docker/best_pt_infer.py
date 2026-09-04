#!/usr/bin/env python3
"""Run the copied best.pt detector on one RGB image."""

import argparse
import json
import os
import time

import cv2
from ultralytics import YOLO


def class_id_for(names, target):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in items:
        if str(class_name) == str(target):
            return int(class_id)
    raise ValueError("target class %r not found in %s" % (target, names))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/models/best.pt")
    parser.add_argument("--target-class", default="package")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("cannot read input image: %s" % args.image)

    started = time.perf_counter()
    model = YOLO(args.model)
    target_id = class_id_for(model.names, args.target_class)
    result = model.predict(
        image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=0,
        classes=[target_id],
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    annotated = result.plot(conf=True, labels=True, boxes=True)
    if not cv2.imwrite(args.output, annotated):
        raise SystemExit("cannot write output image: %s" % args.output)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(model.names[class_id]),
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
                "names": model.names,
                "target_class": args.target_class,
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "elapsed_ms": round(elapsed_ms, 1),
                "detections": detections,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
