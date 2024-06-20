import math

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
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

if True:
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window()
    frames = pipeline.wait_for_frames()
    frames = align_to_color.process(frames)

    # 获取深度图和彩色图
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # 将深度图和彩色图转换为NumPy数组
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 预测目标实例结果
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

    # 膨胀后的实例mask：mask_base_copy
    mask_base_copy = mask_base.copy()
    kernel_size = (15, 15)
    kernel = np.ones(kernel_size, np.uint8)
    mask_base_copy = cv2.dilate(mask_base_copy, kernel, iterations=1)

    # 目标box的mask并且扩大:box_mask_base
    min_x, max_x, min_y, max_y = 640, 0, 480, 0
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy[0]
        min_x = min(min_x, int(xyxy[0]))
        min_y = min(min_y, int(xyxy[1]))
        max_x = max(max_x, int(xyxy[2]))
        max_y = max(max_y, int(xyxy[3]))

    siexl_add = 50
    min_x = min_x - siexl_add
    min_y = min_y - siexl_add
    max_x = max_x + siexl_add
    max_y = max_y + siexl_add
    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0
    if max_x > 639:
        max_x = 639
    if max_y > 479:
        max_y = 479
    top_left = (min_x, min_y)
    bottom_right = (max_x, max_y)

    box_mask_base = np.zeros_like(mask_base_copy)
    box_mask_base[min_y:max_y, min_x:max_x] = 1

    box_mask_base = box_mask_base.reshape(depth_image.shape)
    mask_base_copy = mask_base_copy.reshape(depth_image.shape)
    mask_base = mask_base.reshape(depth_image.shape)
    depth_image_area = depth_image.copy()
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 生成底面点云
    depth_image_area[mask_base_copy == 1] = 0
    depth_image_area[box_mask_base == 0] = 0
    # 彩色图、深度图转open3d格式
    color = o3d.geometry.Image(color_image)
    depth_area = o3d.geometry.Image(depth_image_area)
    # 创建rgb-d-area
    rgbd_image_area = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth_area,
                                                                         convert_rgb_to_intensity=False)
    # 创建点云
    pcd_area = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_area, pinhole_camera_intrinsic)
    visualizer.add_geometry(pcd_area)

    # 生成每个实例点云和一个点云对象列表
    pcd_object_all = o3d.geometry.PointCloud()
    pcd_object_list = []
    center_xy = []
    if result.masks is not None:
        masks = result.masks
        for mask in masks:
            mask_np = mask.data.numpy().astype(np.uint8)
            mask_np = np.transpose(mask_np, (1, 2, 0))
            mask_np = mask_np.reshape(depth_image.shape)
            contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_moment = cv2.moments(contours[0])
            if contours_moment["m00"] != 0:
                c_x = int(contours_moment["m10"] / contours_moment["m00"])
                c_y = int(contours_moment["m01"] / contours_moment["m00"])
            else:
                c_x = 0
                c_y = 0
            center_xy.append([c_x, c_y])

            depth_image_object = depth_image.copy()
            depth_image_object[mask_np == 0] = 0
            depth_object = o3d.geometry.Image(depth_image_object)
            rgbd_image_object = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth_object,
                                                                                   convert_rgb_to_intensity=False)
            pcd_class = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_object, pinhole_camera_intrinsic)
            pcd_object_list.append(pcd_class)
            pcd_object_all += pcd_class
    else:
        mask_base = np.ones((480, 640, 1)).astype(np.uint8)
        print('none detected!')

    visualizer.add_geometry(pcd_object_all)

    # 拟合底面平面
    # 降噪点
    pcd_filtered, _ = pcd_area.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # 使用RANSAC算法拟合平面，设置最大迭代次数和距离阈值
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model

    # # 提取内点和离群点
    # inlier_cloud = pcd_area.select_by_index(inliers)
    #
    # # 创建一个网格来表示平面
    # # 这里我们需要制定平面的一个中心点和平面的一个法向量
    # plane_center = inlier_cloud.get_center()
    # plane_normal = np.array([a, b, c])
    # # Open3D没有直接的API来创建一个平面几何体
    # # 因此，我们创建一个足够大的网格平面，并沿着法向量旋转它，然后平移到中心点
    # mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
    # mesh.translate(-mesh.get_center())
    # R = mesh.get_rotation_matrix_from_xyz((0, 0, np.arctan2(plane_normal[1], plane_normal[0])))
    # mesh.rotate(R, center=(0, 0, 0))
    # mesh.translate(plane_center)
    # # 着色网格平面（例如蓝色）
    # mesh.paint_uniform_color([0.1, 0.1, 0.7])
    # visualizer.add_geometry(mesh)

    # 遍历点云列表中每个点云的每个点计算与拟合平面的最远距离点
    # 遍历点云对象列表
    for i, pcd in enumerate(pcd_object_list):
        average_distance = 0
        max_distance = 0
        # 将点云对象的点坐标转换为Numpy数组
        points = np.asarray(pcd.points)
        # 遍历点坐标
        for point in points:
            # 可以访问每个点的坐标
            x = point[0]
            y = point[1]
            z = point[2]
            distance = math.fabs(a * x + b * y + c * z + d) / (math.sqrt(a * a + b * b + c * c))
            max_distance = max(distance, max_distance)

        real_z = -d / c
        real_z = real_z - max_distance / 2
        real_x = (center_xy[i][0] - intrinsics.ppx) * real_z / intrinsics.fx
        real_y = (center_xy[i][1] - intrinsics.ppy) * real_z / intrinsics.fy

        centroid = np.asarray((real_x, real_y, real_z), dtype='float64')
        radius = 0.005  # 设置球体的半径
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        centroid_sphere.translate(centroid)
        # 将球体颜色更改为红色
        red_color = [1.0, 0.0, 0.0]  # 红色
        centroid_sphere.paint_uniform_color(red_color)
        visualizer.add_geometry(centroid_sphere)
        print(real_x, real_y, real_z)

    visualizer.run()

# 停止并关闭RealSense相机
pipeline.stop()
visualizer.destroy_window()
