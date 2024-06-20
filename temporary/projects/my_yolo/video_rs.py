import cv2
import numpy as np
import pyrealsense2 as rs
import time

from ultralytics import YOLO
# import cv2
# import numpy as np

# 初始化RealSense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始捕获
pipeline.start(config)

# 配置VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('saved_video/output_video_'+str(time.time())+'.avi', fourcc, 30.0, (640, 480))

# Load a model
model = YOLO('pts/best-train3.pt')  # load a custom model

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
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        
        # 将帧转换为OpenCV格式
        image = np.asanyarray(color_frame.get_data())

        cv2.circle(image, (320, 240), 3, (0, 0, 255), 2, 8, 0)

        cv2.imshow("image", image)

        # results = model(image)
        #
        # for result in results:
        #     image = visualize(result)
        #
        #
        # # 显示图像
        # cv2.imshow("RealSense", image)
        #
        # # 保存到视频文件
        # out.write(image)

        # 检查按键，按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 关闭VideoWriter和摄像头
    out.release()
    pipeline.stop()
    cv2.destroyAllWindows()
