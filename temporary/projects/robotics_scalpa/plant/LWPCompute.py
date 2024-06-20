from sklearn.pipeline import Pipeline
import numpy as np
import pcl
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import boundaryDetect as bd
import curveFitting as cf
import curveLengthCompute as clc
import resampling as rs


def compute_extreme_points(points: np.ndarray):

    max_lwh_index = np.argmax(points, axis=0)
    min_lwh_index = np.argmin(points, axis=0)

    pointsT = points.transpose()
    max_value = pointsT[range(len(max_lwh_index)), max_lwh_index]
    min_value = pointsT[range(len(max_lwh_index)), min_lwh_index]
    result = np.abs(max_value - min_value)
    l_index = np.argmax(result)      #the max value is length

    max_l_index = max_lwh_index[l_index]
    min_l_index = min_lwh_index[l_index]

    extreme_points = np.array([points[max_l_index], points[min_l_index]])

    return  max_l_index, min_l_index, extreme_points


def align_points_to_axes(boundary_pc: o3d.geometry.PointCloud):

    obb = boundary_pc.get_oriented_bounding_box()  # 测试时间
    obb.color = (0, 1, 0)

    flag_show = True
    if flag_show:
        o3d.visualization.draw_geometries([boundary_pc, obb], window_name='NAUCVL', width=1920, height=1080, left=0,
                                          top=30)

    rotation_matrix = np.linalg.inv(obb.R)
    boundary_pc.rotate(rotation_matrix)
    #boundary_pc.translate(-obb.center)

    # o3d.io.write_point_cloud("../data/boundary.pcd", boundary_pc)

    return boundary_pc


def compute_perimeter(points_xyz_half,  extreme_points):

    X = points_xyz_half[:, 0:1]
    X = np.array(X, dtype=np.float64)

    Y = points_xyz_half[:, [1]].reshape(points_xyz_half.shape[0])
    poly_reg_A = cf.polynomial_fit(X, Y)
    Y_predict = poly_reg_A.predict(X)

    Z = points_xyz_half[:, [2]].reshape(points_xyz_half.shape[0])
    poly_reg_B = cf.polynomial_fit(X, Z)

    Z_predict = poly_reg_B.predict(X)

    x_range = extreme_points[:, [0]]
    length = clc.curve_3d(x_range, poly_reg_A, poly_reg_B)


    X_row = X.reshape(X.shape[0])

    flag_show = True
    if flag_show:
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        ax = plt.axes(projection='3d')
        #ax = plt.subplots(projection='3d')
        X_index = np.argsort(X_row)

        ax.plot3D(np.sort(X_row), Y_predict[X_index], Z_predict[X_index], 'gray')
        ax.scatter3D(X_row, Y, Z)#, c=Z, cmap='Greens'

        plt.show()
        #plt.title("3-D Curve")



    return length


def len_wid_peri_compute(pc_o3d: o3d.geometry.PointCloud):

    boundary_pc = bd.angle_criterion(pc_o3d)

    flag_show = True
    if flag_show:
        o3d.visualization.draw_geometries([boundary_pc], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

    # boundary_pc = rs.MLS(boundary_pc)
    # if flag_show:
    #    o3d.visualization.draw_geometries([boundary_pc], window_name='NAUCVL', width=1920, height=1080, left=0, top=30)

    # center_point = pc_o3d.get_center()

    boundary_pc = align_points_to_axes(boundary_pc)
    aabb = boundary_pc.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    #obb = boundary_pc.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    if flag_show:
        o3d.visualization.draw_geometries([boundary_pc, aabb], window_name='NAUCVL', width=1920, height=1080, left=0,
                                          top=30)

    box_length = np.abs(aabb.max_bound[0] - aabb.min_bound[0])
    box_width = np.abs(aabb.max_bound[1] - aabb.min_bound[1])
    box_height = np.abs(aabb.max_bound[2] - aabb.min_bound[2])
    print(f'长度为：{box_length}')
    print(f'宽度为：{box_width}')

    points_xyz = np.asarray(boundary_pc.points, dtype='float32')
    max_l_index, min_l_index, extreme_points = compute_extreme_points(points_xyz)
    #leaf_length = np.linalg.norm(extreme_points[0] - extreme_points[1])

    # segment the boundary in half

    points_xy = points_xyz[:, 0:2]
    points_xy1 = np.insert(points_xy, 2, 1., axis=1)

    x = np.array([extreme_points[0][0], extreme_points[1][0]])
    y = np.array([extreme_points[0][1], extreme_points[1][1]])
    slope, intercept = np.polyfit(x, y, 1)

    line_coef = np.array([slope, -1., intercept])
    result = np.dot(line_coef, points_xy1.transpose())

    # np.delete(points_xy1, np.where(a < 2)[0], axis=0)

    curve_length = []
    #half_filter = np.logical_not(result[:] >= 0)
    half_filter = (result >= 0)
    curve_length.append(compute_perimeter(points_xyz[half_filter], extreme_points))
    half_filter = (result <= 0)
    curve_length.append(compute_perimeter(points_xyz[half_filter], extreme_points))

    leaf_perimeter = np.sum(curve_length)
    print(f'周长为：{leaf_perimeter}')

    return box_length, box_width, leaf_perimeter


if __name__ == "__main__":

    pc_o3d = o3d.io.read_point_cloud('../data/boundary.pcd')
    len_wid_peri_compute(pc_o3d)

    pass




