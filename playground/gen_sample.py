import cv2
import numpy as np

def gen_sample_1():
    """生成三个相交的圆形，圆心呈三角形"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # 绘制三个相交的圆
    cv2.circle(image, (150, 150), 80, (0, 0, 0), -1)
    cv2.circle(image, (250, 150), 60, (0, 0, 0), -1)
    cv2.circle(image, (200, 250), 70, (0, 0, 0), -1)

    return image

def gen_sample_2():
    """生成三个相交的圆形，圆心在水平直线上"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # 绘制三个水平排列的相交圆
    cv2.circle(image, (120, 200), 70, (0, 0, 0), -1)
    cv2.circle(image, (200, 200), 75, (0, 0, 0), -1)
    cv2.circle(image, (280, 200), 70, (0, 0, 0), -1)

    return image

def gen_sample_3():
    """生成四个相交的圆形，圆心在矩形的四个角"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # 绘制四个矩形角落的相交圆
    cv2.circle(image, (130, 130), 80, (0, 0, 0), -1)  # 左上角
    cv2.circle(image, (270, 130), 80, (0, 0, 0), -1)  # 右上角
    cv2.circle(image, (130, 270), 80, (0, 0, 0), -1)  # 左下角
    cv2.circle(image, (270, 270), 80, (0, 0, 0), -1)  # 右下角

    return image