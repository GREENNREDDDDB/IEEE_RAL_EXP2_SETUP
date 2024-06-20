import cv2
import os

def extract_frames(input_folder):
    # 获取所有avi文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)

        # 为每个视频创建一个对应的文件夹
        output_folder = os.path.join(input_folder, os.path.splitext(video_file)[0])
        os.makedirs(output_folder, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 获取视频帧率和总帧数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 每隔一定间隔提取一帧
        frame_interval = total_frames // 30

        for i in range(0, total_frames, frame_interval):
            # 设置帧的位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            # 读取帧
            ret, frame = cap.read()

            if ret:
                # 保存帧到新文件夹
                frame_name = f"{os.path.splitext(video_file)[0]}_{i // frame_interval}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)

        # 关闭视频文件
        cap.release()

if __name__ == "__main__":
    root_folder = "./"
    extract_frames(root_folder)
