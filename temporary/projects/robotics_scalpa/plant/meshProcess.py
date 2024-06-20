from numpy.random.mtrand import laplace
import open3d as o3d
import numpy as np


def alpha_shapes():
    bunny_path='../data/KnotMesh.ply'
    mesh = o3d.io.read_triangle_mesh(bunny_path)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(750)
    print("Displaying input pointcloud ...")
    o3d.visualization.draw_geometries([pcd])
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    print('Running alpha shapes surface reconstruction ...')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha)
    mesh.compute_triangle_normals(normalized=True)
    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def ball_pivoting():
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(3000)
    print("Displaying input pointcloud ...")
    o3d.visualization.draw([pcd], point_size=5)

    radii = [0.005, 0.01, 0.02, 0.04]
    print('Running ball pivoting surface reconstruction ...')
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw([rec_mesh])


def poisson():
    eagle = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(eagle.path)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, -np.pi / 4, 0))
    pcd.rotate(R, center=(0, 0, 0))
    print('Displaying input pointcloud ...')
    o3d.visualization.draw([pcd])

    print('Running Poisson surface reconstruction ...')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    print('Displaying reconstructed mesh ...')
    o3d.visualization.draw([mesh])


def create_triangle_mesh(mode:int):
    if mode == 0:
        alpha_shapes()
    elif mode == 1:
        ball_pivoting()
    elif mode == 2:
        poisson()


def average_filtering(mesh_in: o3d.geometry.TriangleMesh):

    #vertices = np.asarray(mesh_in.vertices)
    #noise = 5
    #vertices += np.random.uniform(0, noise, size=vertices.shape)
    #mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
    #mesh_in.compute_vertex_normals()
    #print("Displaying input mesh ...")
    #o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of average mesh filter after 1 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    return mesh_out


def laplace_filtering(mesh_in: o3d.geometry.TriangleMesh):

    print("Displaying output of Laplace mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    return mesh_out


def taubin_filtering(mesh_in: o3d.geometry.TriangleMesh):

    print("Displaying output of Taubin mesh filter after 100 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
    mesh_out.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh_out])

    return mesh_out


def triangle_mesh_filtering(mesh_in: o3d.geometry.TriangleMesh,mode: int):
    if mode == 0:
        mesh_out = average_filtering(mesh_in)
    elif mode == 1:
        mesh_out = laplace_filtering(mesh_in)
    elif mode == 2:
        mesh_out = taubin_filtering(mesh_in)

    return mesh_out

if __name__ == "__main__":
    alpha_shapes()