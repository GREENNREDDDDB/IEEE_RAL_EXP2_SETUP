import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import time

from VisualYolo import VisualYolo

from d435_driver import D435

from ultralytics import YOLO

MODEL_PATH = 'pts/best-train3.pt'

try:

    d435 = D435()
    # Load a model
    model = YOLO(MODEL_PATH)  # load a custom model

    intrinsics = d435.getIntrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    # 创建点云的可视化窗口
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    while True:
        color_image, depth_image = d435.getColorAndDepthImage()
        # results = model.predict(color_image, conf=0.6)
        # for result in results:
        #     color_image = visualize(result)
        # cv2.imshow("RealSense", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    D435.pipeline.stop()
