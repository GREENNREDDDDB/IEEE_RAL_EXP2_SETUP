import cv2
import numpy as np


class VisualYolo:

    def __init__(self, result):
        self.result = result
        self.result.orig_img = result.orig_img.copy()  # copy one
        self.centers = None  # 质心点坐标列表
        self.contours_list = None
        self.texts = None
        self.boxes = None

    def getBoxes(self):
        if self.result.masks is None:
            return None
        if self.boxes is not None:
            return self.boxes

        self.boxes = []
        self.texts = []

        boxes = self.result.boxes  # 假设xyxy属性包含框的坐标信息
        names = self.result.names
        image = self.result.orig_img

        for box in boxes:
            cls_id = int(box.cls)
            cls_name = str(names[cls_id])
            conf = round(float(box.conf), 2)

            xyxy = box.xyxy[0]  # 将xyxy变成长度为4的一维数组
            top_left = (int(xyxy[0]), int(xyxy[1]))
            bottom_right = (int(xyxy[2]), int(xyxy[3]))

            self.boxes.append((top_left, bottom_right))  # add one box
            self.texts.append(cls_name + ' ' + str(conf))
            # cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
            # put text

    def getCenters(self):  # 输入为H W 1 形状的掩码图， 输出第一个轮廓线的质心坐标
        if self.result is None or self.result.masks is None:
            return None
        if self.centers is not None:
            return self.centers
        # 尝试开始画质心点
        self.centers = []
        self.contours_list = []
        for mask_obj in self.result.masks:

            mask = mask_obj.data.numpy().astype(np.uint8)
            mask = np.transpose(mask, (1, 2, 0))
            mask[mask > 0] = 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contours_list.append(contours)

            contours_moment = cv2.moments(contours[0])
            if contours_moment["m00"] != 0:
                c_x = int(contours_moment["m10"] / contours_moment["m00"])
                c_y = int(contours_moment["m01"] / contours_moment["m00"])
                self.centers.append((c_x, c_y))

    def visualize(self):
        image = self.result.orig_img

        self.getBoxes()
        for box, text in zip(self.boxes, self.texts):
            top_left, bottom_right = box
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

            text_position = (top_left[0], top_left[1] - 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.8
            font_thickness = 2
            cv2.putText(image, text, text_position, font, font_size, (0, 0, 255), font_thickness)

        # cv2.putText(image, text, text_position, font, font_size, (0, 0, 255), font_thickness)

        self.getCenters()
        for center, contours in zip(self.centers, self.contours_list):
            cv2.circle(image, center, 5, (255, 0, 0), -1)
            cv2.drawContours(image, contours, 0, (0, 255, 0), 2)
            # 在contour_image上绘制轮廓，颜色为(0, 255, 0)，线宽为2

        return image
