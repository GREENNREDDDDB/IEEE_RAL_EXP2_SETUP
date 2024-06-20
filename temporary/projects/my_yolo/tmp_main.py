from d435_driver import D435

import cv2
import numpy as np
import pyrealsense2 as rs
import time

from ultralytics import YOLO


def visualize(result):
    if result.masks is None:
        return result.orig_img
    boxes = result.boxes  # 假设xyxy属性包含框的坐标信息
    masks = result.masks  # 假设masks属性包含分割掩码信息
    names = result.names
    image = result.orig_img  # 原始图像

    for box, mask in zip(boxes, masks):
        cls_id = int(box.cls)
        cls_name = str(names[cls_id])
        conf = round(float(box.conf), 2)
        if conf < 0.6:
            continue
        xyxy = box.xyxy[0]
        top_left = (int(xyxy[0]), int(xyxy[1]))
        bottom_right = (int(xyxy[2]), int(xyxy[3]))

        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

        # put text
        text = cls_name + ' ' + str(conf)
        text_position = (int(xyxy[0]), int(xyxy[1]) - 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.8
        font_thickness = 2
        cv2.putText(image, text, text_position, font, font_size, (0, 0, 255), font_thickness)

        mask = mask.data.numpy().astype(np.uint8)
        mask = np.transpose(mask, (1, 2, 0))
        mask[mask > 0] = 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, 0, (0, 255, 0), 2)  # 在contour_image上绘制轮廓，颜色为(0, 255, 0)，线宽为2

    return image

try:
     
    d435 = D435()
    # Load a model
    model = YOLO('pts/best-train3.pt')  # load a custom model
    
    while True:
        image, _ = d435.getColorAndDepthImage()
        results = model(image)
        for result in results:
            image = visualize(result)
        cv2.imshow("RealSense", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    D435.pipeline.stop()
    