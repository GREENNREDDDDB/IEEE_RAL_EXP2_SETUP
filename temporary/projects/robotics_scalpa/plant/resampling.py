# -*- coding: utf-8 -*-
# Smoothing and normal estimation based on polynomial reconstruction
# http://pointclouds.org/documentation/tutorials/resampling.php#moving-least-squares

import numpy as np
import pcl
import random
import open3d as o3d

def MLS(pc_o3d: o3d.geometry.PointCloud):

    #cloud = pcl.load('D:/Projects/VS/test_ volume2/test_ volume2/downsampled/downsampled1.pcd')
    pc_pcl = pcl.PointCloud()

    points = np.asarray(pc_o3d.points,dtype='float32')
    pc_pcl.from_array(points)

    print('cloud(size) before MLS = ' + str(pc_pcl.size))

    tree = pc_pcl.make_kdtree()
    mls = pc_pcl.make_moving_least_squares()
    # print('make_moving_least_squares')
    mls.set_Compute_Normals(True)
    mls.set_polynomial_fit(True)
    mls.set_Search_Method(tree)
    mls.set_search_radius(7)
    mls_pc = mls.process()

    print('cloud(size) after MLS = ' + str(mls_pc.size))

    points = mls_pc.to_array()
    pc_o3d.points = o3d.utility.Vector3dVector(points)
    #pcl.save_PointNormal(mls_points, 'bun0-mls.pcd')

    return pc_o3d

