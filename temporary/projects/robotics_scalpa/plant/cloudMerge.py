import open3d as o3d
import numpy as np


for i in range(1, 4):
    pc = o3d.io.read_point_cloud("../data/plantData/A%d.pcd" % i)
    pc = pc.voxel_down_sample(voxel_size=1.5)
    print(pc)
    points = np.asarray(pc.points, dtype='float32')
    points_sum = np.append(points)
pc_sum = o3d.geometry.PointCloud()
pc_sum.points = o3d.utility.Vector3dVector(points_sum)
print(pc_sum)

