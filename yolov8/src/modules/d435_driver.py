import cv2
import numpy as np
import pyrealsense2 as rs


class D435:
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置RealSense相机
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 启动RealSense相机
    # pipeline.start(config)
    # 启动RealSense相机
    profile = pipeline.start(config)

    def __init__(self):
        # 等待相机稳定
        for i in range(30):
            D435.pipeline.wait_for_frames()

    def getColorAndDepthImage(self):

        # 对齐到RGB的相机坐标系
        align_to_color = rs.align(rs.stream.color)
        frames = D435.pipeline.wait_for_frames()
        frames = align_to_color.process(frames)

        # 获取深度图和彩色图
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("D435获取图像失败")
            return None, None
        # 将深度图和彩色图转换为NumPy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # print('return 2 args')
        return color_image, depth_image

    def getIntrinsics(self):
        # 获得内参
        intrinsics = D435.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return intrinsics

# d435 = D435()
# image, _ = d435.getColorAndDepthImage()
# cv2.imshow("Color Image1", image)
# cv2.waitKey(0)

# d435 = D435()
# image, _ = d435.getColorAndDepthImage()
# cv2.imshow("Color Image2", image)
# cv2.waitKey(0)

# D435.pipeline.stop()
