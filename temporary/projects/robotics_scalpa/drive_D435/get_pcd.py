import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

def get_point_cloud(name: str):

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    print(depth_profile)
    depth_intrinsics = depth_profile.get_intrinsics()
    print(depth_intrinsics)

    intrinsicsDC = rs.intrinsics()
    intrinsicsDC.width = depth_intrinsics.width
    intrinsicsDC.height = depth_intrinsics.height
    intrinsicsDC.ppx = depth_intrinsics.ppx
    intrinsicsDC.ppy = depth_intrinsics.ppy
    intrinsicsDC.fx = depth_intrinsics.fx
    intrinsicsDC.fy = depth_intrinsics.fy

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    i = 0
    # Streaming loop
    try:
        while i < 15:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Render images:
            #   depth align to color on left
            #   depth on right
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            i = i + 1

            # plt.imshow(img3, 'gray')
            # plt.show()

        cv2.imwrite("/home/njau/projects/robotics_scalpa/data/depth/" + name + ".png", depth_image)
        cv2.imwrite("/home/njau/projects/robotics_scalpa/data/color/" + name + ".png", color_image)

        points = np.zeros((depth_image.shape[0] * depth_image.shape[1], 3), dtype=np.float64)
        colors = np.zeros((color_image.shape[0] * color_image.shape[1], 3), dtype=np.float64)

        for r in range(depth_image.shape[0]):
            for c in range(depth_image.shape[1]):
                # Pass through filter
                if depth_image[r][c] > 0 and depth_image[r][c] < 500:
                    x, y, z = rs.rs2_deproject_pixel_to_point(intrinsicsDC, [r, c], depth_image[r, c])
                    idx = r * depth_image.shape[1] + c
                    points[idx, 0] = np.float64(x)
                    points[idx, 1] = np.float64(y)
                    points[idx, 2] = np.float64(z)
                    colors[idx, 0] = np.float64(color_image[r, c][2] / 255)
                    colors[idx, 1] = np.float64(color_image[r, c][1] / 255)
                    colors[idx, 2] = np.float64(color_image[r, c][0] / 255)
        mask = (points[:, 2] != 0) & (points[:, 2] <= 500)
        points = points[mask]
        colors = colors[mask]
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(points)
        pc_o3d.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pc_o3d])
        file_path = "/home/njau/projects/robotics_scalpa/data/pcd/" + name + ".pcd"
        o3d.io.write_point_cloud(file_path, pc_o3d)
        print('save done')

    finally:
        pipeline.stop()

    return pc_o3d
