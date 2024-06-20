import open3d as o3d
import numpy as np
import cv2
import pyrealsense2 as rs

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动RealSense相机
profile = pipeline.start(config)

# intrinsics
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

try:
    while True:
        # 等待帧
        #align to color
        align_to_color = rs.align(rs.stream.color)  
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)

        # 获取深度图和彩色图
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 将深度图和彩色图转换为NumPy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image_normalized = cv2.normalize(color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        color = o3d.geometry.Image(color_image_normalized)
        depth = o3d.geometry.Image(depth_image.astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth)


        # 创建点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
            )
        )
        pcd.colors = o3d.utility.Vector3dVector(color_image_normalized.reshape(-1, 3))
        # 创建一个可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 将点云添加到可视化窗口
        vis.add_geometry(pcd)

        # 设置视角并渲染可视化窗口
        vis.get_render_option().point_size = 1

        # 运行可视化窗口
        vis.run()
        vis.destroy_window()
        if cv2.waitKey(1) == 27:
            break
finally:
    # 停止并关闭RealSense相机
    pipeline.stop()
