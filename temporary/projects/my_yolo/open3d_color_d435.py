import open3d as o3d
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO


model = YOLO('pts/best-train3.pt')  # load a custom model

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动RealSense相机
profile = pipeline.start(config)

# 获得内参
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 等待曝光
for _ in range(30):
    pipeline.wait_for_frames()

# 对齐至彩色图位置
align_to_color = rs.align(rs.stream.color)

# 内参转open3d格式内参
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)


# 创建可视化窗口
# visualizer = o3d.visualization.Visualizer()
# visualizer.create_window()

if True:
    frames = pipeline.wait_for_frames()
    frames = align_to_color.process(frames)

    # 获取深度图和彩色图
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # 将深度图和彩色图转换为NumPy数组
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    results = model.predict(color_image, conf=0.6)
    result = results[0]
    mask_base = np.zeros((480, 640, 1)).astype(np.uint8)
    if result.masks is not None:
        masks = result.masks
        for mask in masks:
            mask_np = mask.data.numpy().astype(np.uint8)
            mask_np = np.transpose(mask_np, (1, 2, 0))
            mask_base = mask_base | mask_np
    else:
        mask_base = np.ones((480, 640, 1)).astype(np.uint8)
        print('none detected!')


    # cv2.imshow("mask_base", mask_base)


    # color_image = color_image & mask_base
    # depth_image = depth_image & mask_base.reshape(depth_image.shape)
    mask_base = mask_base.reshape(depth_image.shape)
    depth_image[mask_base == 0] = 0
    #cv2.imshow("color_image", color_image)

    #cv2.waitKey(0)

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


    # 彩色图、深度图转open3d格式
    color = o3d.geometry.Image(color_image)
    depth = o3d.geometry.Image(depth_image)

    # 创建rgb-d
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)

    # 创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)

    o3d.visualization.draw_geometries([pcd])

    # 更新可视化窗口
    # visualizer.clear_geometries()
    # visualizer.add_geometry(pcd)
    #visualizer.update_geometry(pcd)
    #visualizer.poll_events()
    #visualizer.update_renderer()

# 停止并关闭RealSense相机
pipeline.stop()
# visualizer.destroy_window()
