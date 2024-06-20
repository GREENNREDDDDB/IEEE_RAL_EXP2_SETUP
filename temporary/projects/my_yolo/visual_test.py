import cv2
import numpy as np
import pyrealsense2 as rs
import time

from VisualYolo import VisualYolo
from d435_driver import D435

from ultralytics import YOLO

MODEL_PATH = 'pts/best-train3.pt'

try:

    d435 = D435()
    # Load a model
    model = YOLO(MODEL_PATH)  # load a custom model

    while True:
        image, _ = d435.getColorAndDepthImage()
        results = model.predict(image, conf=0.6)
        visualYolo = VisualYolo(results[0])
        for result in results:
            image = visualYolo.visualize()
        cv2.imshow("RealSense", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    D435.pipeline.stop()
