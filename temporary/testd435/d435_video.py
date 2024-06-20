import pyrealsense2 as rs
import numpy as np
import cv2


class box_video_data:
    def __init__(self):
        self.dev = None
        self.intrinsics = None
        self.intrinsics2 = ""
        self.color_images = None
        self.depth_images = None


class BeanBox:
    def __init__(self):
        pass

    def rsintrinsics2string(self, intrinsics):
        intrinsics_str = ""
        intrinsics_str += f"{intrinsics.width}/"
        intrinsics_str += f"{intrinsics.height}/"
        intrinsics_str += f"{intrinsics.fx}/"
        intrinsics_str += f"{intrinsics.fy}/"
        intrinsics_str += f"{intrinsics.ppx}/"
        intrinsics_str += f"{intrinsics.ppy}/"
        intrinsics_str += f"{intrinsics.coeffs[0]}/"
        intrinsics_str += f"{intrinsics.coeffs[1]}/"
        intrinsics_str += f"{intrinsics.coeffs[2]}/"
        intrinsics_str += f"{intrinsics.coeffs[3]}/"
        intrinsics_str += f"{intrinsics.coeffs[4]}/"
        return intrinsics_str

    def string2rsintrinsics(self, intrinsics_str):
        intrinsics = rs.intrinsics()
        tokens = intrinsics_str.split('/')
        intrinsics.width = int(tokens[0])
        intrinsics.height = int(tokens[1])
        intrinsics.fx = float(tokens[2])
        intrinsics.fy = float(tokens[3])
        intrinsics.ppx = float(tokens[4])
        intrinsics.ppy = float(tokens[5])
        intrinsics.coeffs[0] = float(tokens[6])
        intrinsics.coeffs[1] = float(tokens[7])
        intrinsics.coeffs[2] = float(tokens[8])
        intrinsics.coeffs[3] = float(tokens[9])
        intrinsics.coeffs[4] = float(tokens[10])
        return intrinsics

    def d435_pre(self, d435_video, dev, i):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = pipe.start(cfg)


        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        d435_video[i].dev = dev
        d435_video[i].intrinsics = intrinsics
        intrinsics_str = self.rsintrinsics2string(intrinsics)
        d435_video[i].intrinsics2 = intrinsics_str

        # print(d435_video[i].intrinsics2)

        # pipe.stop()
        for _ in range(30):
            pipe.wait_for_frames()  # Drop several frames for auto-exposure

        return pipe

    def get_d435_image(self,d435_video,pipe):

        align_to_color = rs.align(rs.stream.color)

        frameset = pipe.wait_for_frames()  # Drop several frames for auto-exposure
        frameset = align_to_color.process(frameset)  # Align depth frame to color frame

        depth_frame = frameset.get_depth_frame()  # Get depth frame
        color_frame = frameset.get_color_frame()  # Get color frame

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        depth_image = np.asanyarray(depth_frame.get_data())
        
        color_image = cv2.flip(color_image, -1)
        depth_image = cv2.flip(depth_image, -1)

        d435_video.color_images = color_image
        d435_video.depth_images = depth_image

    def close_d435(self,pipe):
        pipe.stop()

    def run_box(self):

        # 获取设备号，并按照设备数量，新建设备数据类
        ctx = rs.context()
        devices = ctx.query_devices()
        count = len(devices)
        video = [box_video_data() for _ in range(count)]


        # 启动设备并获取内参
        # for i in range(count):
        #     dev = devices[i]
        #     self.d435_pre(video, dev, i)

        # 启动设备0，并做预曝光处理，以及获取内参
        pipe = self.d435_pre(video, devices[0], 0)

        # 。。。。。。。可以插入获取图片数据的信号代码
        # 。。。。。。。可以插入获取图片数据的信号代码
        # 。。。。。。。可以插入获取图片数据的信号代码
        # 。。。。。。。可以插入获取图片数据的信号代码
        # 。。。。。。。可以插入获取图片数据的信号代码
        # 。。。。。。。可以插入获取图片数据的信号代码
        # 。。。。。。。可以插入获取图片数据的信号代码

        # 获得相机0的彩色图片和深度图片
        self.get_d435_image(video[0], pipe)

        # 关闭相机0
        self.close_d435(pipe)

        # 返回相机0获得的数据
        return video[0]



if __name__ == '__main__':
    box = BeanBox()

    video = box.run_box()
    # print(video.intrinsics2)
    # cv2.imwrite("./image_data/color_1.png", video.color_images)
    # cv2.imwrite("./image_data/depth_1.png", video.depth_images)
