import open3d as o3d
import numpy as np
import cv2
import math


class PointCloudController:

    def __init__(self):
        self.visualizer = o3d.visualization.Visualizer()
        self.intrinsics_o3d = None
        self.instance_pcd_list = []
        self.box_other_pcd = None
        # self.visualizer.create_window()

    def setIntrinsic(self, intrinsics_rs):
        # 内参转open3d格式内参
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            intrinsics_rs.width, intrinsics_rs.height, intrinsics_rs.fx,
            intrinsics_rs.fy, intrinsics_rs.ppx, intrinsics_rs.ppy)

    def unionBoxes(self, resultYolo, padding: int = 50):
        if resultYolo.boxes is None:
            return None, None

        H, W = resultYolo.orig_img.shape[:2]

        min_x, max_x, min_y, max_y = W, 0, H, 0

        for box in resultYolo.boxes:
            xyxy = box.xyxy[0]
            min_x = min(min_x, int(xyxy[0]))
            min_y = min(min_y, int(xyxy[1]))
            max_x = max(max_x, int(xyxy[2]))
            max_y = max(max_y, int(xyxy[3]))

        min_x = min_x - padding
        min_y = min_y - padding
        max_x = max_x + padding
        max_y = max_y + padding
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > W - 1:
            max_x = W - 1
        if max_y > H - 1:
            max_y = H - 1

        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        return top_left, bottom_right

    def genaratePcd(self, color_image, depth_image, mask):
        mask = mask.reshape(depth_image.shape)

        depth_image = depth_image.copy()
        depth_image[mask == 0] = 0

        depth_image_o3d = o3d.geometry.Image(depth_image)
        color_image_o3d = o3d.geometry.Image(color_image)
        # 创建rgb-d-area
        rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d,
                                                                            convert_rgb_to_intensity=False)

        # o3d.visualization.draw_geometries([pcd])
        # 从RGB-D图像创建点云
        depth_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image_o3d,
            self.intrinsics_o3d
        )
        return depth_pcd

    def fitPlane(self, resultYolo, depth_image):
        image = resultYolo.orig_img
        if resultYolo.masks is None:
            return None, None, None, None
        mask_base = np.zeros((image.shape[0], image.shape[1], 1)).astype(np.uint8)
        for mask in resultYolo.masks:
            mask_np = mask.data.numpy().astype(np.uint8)
            mask_np = np.transpose(mask_np, (1, 2, 0))
            mask_base |= mask_np
            self.instance_pcd_list.append(self.genaratePcd(image, depth_image, mask_np))

        # mask_base 是所有识别到物体的掩码的并集
        top_left, bottom_right = self.unionBoxes(resultYolo, 50)
        if top_left is None or bottom_right is None:
            return None, None, None, None
        min_x, min_y = top_left
        max_x, max_y = bottom_right

        # 膨胀后的实例mask: mask_base_copy
        mask_base_copy = mask_base.copy()
        kernel_size = (15, 15)
        kernel = np.ones(kernel_size, np.uint8)
        mask_base_copy = cv2.dilate(mask_base_copy, kernel, iterations=1)

        mask_base_copy = mask_base_copy.reshape(depth_image.shape)

        box_mask_base = np.zeros_like(mask_base_copy)
        # box_mask_base[min_x:max_x, min_y:max_y] = 1 !
        box_mask_base[min_y:max_y, min_x:max_x] = 1
        box_mask_base = box_mask_base.reshape(depth_image.shape)

        # 挖去内部的圆洞
        depth_image[mask_base_copy == 1] = 0
        # 只保留box区域
        depth_image[box_mask_base == 0] = 0

        # 创建o3d图像对象
        depth_image_o3d = o3d.geometry.Image(depth_image)
        color_image_o3d = o3d.geometry.Image(image)
        # 创建rgb-d-area
        rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d,
                                                                            convert_rgb_to_intensity=False)

        # o3d.visualization.draw_geometries([pcd])

        # 从RGB-D图像创建点云
        depth_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image_o3d,
            self.intrinsics_o3d
        )

        # 拟合底面平面
        # 降噪点
        pcd_filtered, _ = depth_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # 使用RANSAC算法拟合平面，设置最大迭代次数和距离阈值
        plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.01,
                                                          ransac_n=3, num_iterations=1000)
        # 得到平面方程
        # a, b, c, d = plane_model
        self.box_other_pcd = depth_pcd
        return plane_model

    def visualPointCloud(self, resultYolo, depth_image):
        plane_model = self.fitPlane(resultYolo, depth_image.copy())
        self.visualizer.create_window()
        self.visualizer.add_geometry(self.box_other_pcd)
        # self.visualizer.run()

    def getCentroids(self, plane_model, centers, intrinsics_rs):

        flag = False

        centroids = []
        a, b, c, d = plane_model
        for center, instance_pcd in zip(centers, self.instance_pcd_list):
            max_distance = 0
            for point in np.asarray(instance_pcd.points):
                x, y, z = point[:3]
                distance = math.fabs(a * x + b * y + c * z + d) / (math.sqrt(a * a + b * b + c * c))
                max_distance = max(distance, max_distance)

            real_z = -d / c
            real_z = real_z - max_distance / 2
            real_x = (center[0] - intrinsics_rs.ppx) * real_z / intrinsics_rs.fx
            real_y = (center[1] - intrinsics_rs.ppy) * real_z / intrinsics_rs.fy

            centroid = np.asarray((real_x, real_y, real_z), dtype='float64')
            centroids.append(centroid)

            if flag:
                radius = 0.005  # 设置球体的半径
                centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                centroid_sphere.translate(centroid)
                # 将球体颜色更改为红色
                red_color = [1.0, 0.0, 0.0]  # 红色
                centroid_sphere.paint_uniform_color(red_color)
                self.visualizer.add_geometry(centroid_sphere)
                self.visualizer.add_geometry(instance_pcd)

        if flag:
            self.visualizer.run()
        return centroids
