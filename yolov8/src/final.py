from modules.PointCloudController import PointCloudController
from modules.d435_driver import D435
from ultralytics import YOLO
from modules.VisualYolo import VisualYolo
import cv2
import time


def main():

    model = YOLO('IEEE_RAL_EXP2_SETUP/yolov8/weights/best.pt')  # load a custom model
    d435 = D435()

    intrinsics_rs = d435.getIntrinsics()
    pcc = PointCloudController()
    pcc.setIntrinsic(intrinsics_rs)

    color_image, depth_image = d435.getColorAndDepthImage()
    results = model.predict(color_image, conf=0.45)
    
    visualYolo = VisualYolo(results[0])
    image = visualYolo.visualize()
    print(image)
    # cv2.imshow("visual yolo", image)
    # cv2.waitKey(0)

    # pcc.visualPointCloud(results[0], depth_image)

    plane_model = pcc.fitPlane(results[0], depth_image)
    centroids = pcc.getCentroids(plane_model, visualYolo.centers, intrinsics_rs)

    return centroids

if __name__ == "__main__":
    print(main())
    