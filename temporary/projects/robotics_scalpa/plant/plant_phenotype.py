import cv2
import numpy as np
import pcl
from matplotlib import pyplot as plt
import open3d as o3d
from scipy import stats
import depth2pointCloud as dpc
import meshProcess as mp
import resampling as rs
import LWPCompute as lc

rgb_path = r'..\data\plantSet\color\plant_004.png'
depth_path = r'..\data\plantSet\depth\plant_004.png'
mask_path = r'..\data\plantSet\masks\plant_004\sub_9.png'

depth= cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

depth = cv2.bitwise_and(depth, depth, mask=mask)

pc = pcl.PointCloud()
#points = np.zeros((depth.shape[0] * depth.shape[1], 3), dtype=np.float32)
colors = np.zeros((depth.shape[0] * depth.shape[1], 3), dtype=np.uint8)

depthArr = np.asarray(depth, dtype='float32')      # int16 to float32
depthArr = (depthArr - 32768)
valueZ = depthArr.flatten()

# 过滤奇异值和突出值
valueZ = valueZ[valueZ > 0]
valueZ = valueZ[valueZ < 5000]

split_num = int((valueZ.max() - valueZ.min()) / 3) + 1

bins = np.linspace(valueZ.min(), valueZ.max(), split_num)
#plt.xlim([valueZ.min() - 3, valueZ.max() + 3])

hist = plt.hist(valueZ, bins=bins)
#plt.clf()

histX_left = hist[1][hist[1] < valueZ.mean()]
histX_right = hist[1][hist[1] > valueZ.mean()]

histY_left = hist[0][0:histX_left.size]
histY_right = hist[0][histX_left.size:hist[1].size]

histX_left = histX_left[::-1]
histY_left = histY_left[::-1]

thresh_left = 0
thresh_right = 0

for idx, num in enumerate(histY_left):
    thresh_left = histX_left[idx]
    if num < 10:
        #thresh_left = histX_left[idx]
        break

for idx, num in enumerate(histY_right):
    thresh_right = histX_right[idx]
    if num < 10:
        #thresh_right = histX_right[idx]
        break

# for x in np.nditer(depthArr, op_flags=['readwrite']):
#     if x < thresh_left or x > thresh_right:
#         x[...] = 0.0

points = dpc.depth_to_points_ZED(depthArr, mask)     #convert depth mat captured by ZED camera to points

pass_through_filter = np.logical_and(points[:,2] >= thresh_left ,points[:,2] <= thresh_right)
points=points[pass_through_filter]


# pc_o3d.colors = o3d.utility.Vector3dVector(colors[pass_through_filter])

# pc.from_array(points)
# pcl.save(pc, r'D:\Projects\python\plant_phenotype\lib\tmp.pcd')

# pc_o3d = o3d.io.read_point_cloud(r'D:\Projects\python\plant_phenotype\lib\tmp.pcd')

pc_o3d = o3d.geometry.PointCloud()
pc_o3d.points = o3d.utility.Vector3dVector(points)

flag_show = False

if flag_show:
    o3d.visualization.draw_geometries([pc_o3d], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

# voxel_down_pc = pc_o3d.voxel_down_sample(voxel_size=0.002)
# o3d.visualization.draw_geometries([voxel_down_pc], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

# uni_down_pcd = pc_o3d.uniform_down_sample(every_k_points=5)
# o3d.visualization.draw_geometries([uni_down_pcd], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

cl ,ind = pc_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
cl ,ind = cl.remove_radius_outlier(nb_points=16, radius=5)

if flag_show:
    o3d.visualization.draw_geometries([cl], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

cl = rs.MLS(cl)
if flag_show:
    o3d.visualization.draw_geometries([cl], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

pc_o3d_voxel_down = cl.voxel_down_sample(voxel_size=1.5)
if flag_show:
    o3d.visualization.draw_geometries([pc_o3d_voxel_down], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

o3d.io.write_point_cloud("../data/plant.pcd", pc_o3d_voxel_down)
lc.len_wid_peri_compute(pc_o3d_voxel_down)

#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc_o3d_voxel_down, 100)

#compute the area of leaf

pc_o3d_voxel_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
radii = [0.1, 0.6, 1.2, 1.8]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
   pc_o3d_voxel_down, o3d.utility.DoubleVector(radii))

mesh = mp.triangle_mesh_filtering(mesh,2)

area = mesh.get_surface_area()  # 计算表面积
print("表面积为：", area)

if flag_show:
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

mesh.compute_vertex_normals()
pc_o3d_poisson = mesh.sample_points_poisson_disk(3000)
if flag_show:
    o3d.visualization.draw_geometries([pc_o3d_poisson], point_show_normal=True, window_name='NAUCVL', width=1920, height=1080, left=0, top=30)


normals = np.asarray(pc_o3d_poisson.normals)
normals = normals[normals[:,2] > 0]

normal = np.zeros(3, dtype=np.float64)
for i in range(3):
    normal[i] = np.mean(normals[:,i])

radianXOZ = np.arctan(normal[2] / normal[0])
radianYOZ = np.arctan(normal[2] / normal[1])
angleXOZ = radianXOZ * 180 / np.pi
angleYOZ = radianYOZ * 180 / np.pi

if angleXOZ > 50 or angleYOZ:
    pass

points = np.asarray(pc_o3d_poisson.points)
points = np.around(points, 1)
depthUsed = stats.mode(points[:,2])

#计算每个像素在法向量旋转加权下的单位面积
#每个像素的步长为：
fx = 1387.484130859375
lengthPixelStep = depthUsed.mode[0] / fx
# 法向量 normal 平面 与 XOZ 平面 切线段长度
sideX = lengthPixelStep * np.abs(normal[0] / normal[2]) / np.cos(np.arctan(np.abs(normal[2] / normal[0])))
# 法向量 normal 平面 与 YOZ 平面 切线段长度
sideY = lengthPixelStep * np.abs(normal[1] / normal[2]) / np.cos(np.arctan(np.abs(normal[2] / normal[1])))
# 对角线长度
diagonal = lengthPixelStep * np.sqrt(2.0 + np.square((normal[0] - normal[1]) / normal[2]))
# 海伦公式求解面积：
p = (sideX + sideY + diagonal) / 2
areaPixel = np.sqrt(p * (p - sideX) * (p
                                       - sideY) * (p - diagonal)) * 2
areaComputed = areaPixel * np.count_nonzero(mask)

print(f'计算的面积为：{areaComputed}')

# to PCL end
# pcl.save(pc, r'D:\Projects\python\plant_phenotype\lib\test_depth_filter.pcd')
# o3d.visualization.draw_geometries([pc1, mesh], point_show_normal=True, window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

