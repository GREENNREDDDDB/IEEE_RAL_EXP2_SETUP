import open3d as o3d
import numpy as np

def transform(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud, tr_matrix: np.array):

    # source_pcd = o3d.io.read_point_cloud("../data/plantData/A1.pcd")
    # target_pcd = o3d.io.read_point_cloud("../data/plantData/A1.pcd")

    # 将源点云应用变换矩阵
    trans_pcd = source_pcd.transform(tr_matrix)

    # 融合点云jointaxes
    merged_pcd = target_pcd + trans_pcd

    # 可视化结果
    o3d.visualization.draw_geometries([merged_pcd])

    return
