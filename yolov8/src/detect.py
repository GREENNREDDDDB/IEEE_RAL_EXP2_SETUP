import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from modules.d435_driver import D435


try:
    d435 = D435()
    # Load a model
    model = YOLO('IEEE_RAL_EXP2_SETUP/yolov8/weights/best.pt')  # load a custom model

    while True:
        color_image, depth_image = d435.getColorAndDepthImage()
        # Run YOLOv8 inference on the frame
        results = model(color_image, conf=0.8 )
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # for result in results:
            # image = visualize(result)
        # cv2.imshow("RealSense", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    D435.pipeline.stop()

# # Load the YOLOv8 model
# model = YOLO("ieee_ral_exp2_best.pt")

# # Open the video file
# # video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(2)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################


# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)
# model = YOLO("ieee_ral_exp2_best.pt")

# try:
#     while True:

#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         depth_colormap_dim = depth_colormap.shape
#         color_colormap_dim = color_image.shape
        
#         # Run YOLOv8 inference on the frame
#         results = model(color_image)

#         # Visualize the results on the frame
#         color_image = results[0].plot()

#         # If depth and color resolutions are different, resize color image to match depth image for display
#         if depth_colormap_dim != color_colormap_dim:
#             resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#             images = np.hstack((resized_color_image, depth_colormap))
#         else:
#             images = np.hstack((color_image, depth_colormap))

#         # Show images
#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', images)
#         key = cv2.waitKey(1)
#         # Press esc or 'q' to close the image window
#         if key & 0xFF == ord('q') or key == 27:
#             cv2.destroyAllWindows()
#             break

# finally:

#     # Stop streaming
#     pipeline.stop()