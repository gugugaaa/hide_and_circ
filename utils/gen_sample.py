import cv2
import numpy as np

def gen_sample_1():
    """两个相交圆和一个独立小圆"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.circle(image, (130, 165), 56, (0, 0, 0), -1)
    cv2.circle(image, (229, 169), 84, (0, 0, 0), -1)
    cv2.circle(image, (121, 313), 53, (0, 0, 0), -1)
    return image

def gen_sample_2():
    """三个圆，都相交，圆心呈三角形，中间有空洞"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.circle(image, (126, 225), 70, (0, 0, 0), -1)
    cv2.circle(image, (271, 238), 87, (0, 0, 0), -1)
    cv2.circle(image, (203, 98), 89, (0, 0, 0), -1)
    return image

def gen_sample_3():
    """四个圆，2大2小，圆心在正方形顶点"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.circle(image, (135, 127), 75, (0, 0, 0), -1)
    cv2.circle(image, (234, 122), 42, (0, 0, 0), -1)
    cv2.circle(image, (132, 213), 39, (0, 0, 0), -1)
    cv2.circle(image, (241, 225), 122, (0, 0, 0), -1)
    return image

def gen_sample_4():
    """三个圆，A与BC相交，米老鼠脑袋形状"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.circle(image, (203, 221), 136, (0, 0, 0), -1)
    cv2.circle(image, (103, 132), 66, (0, 0, 0), -1)
    cv2.circle(image, (298, 107), 40, (0, 0, 0), -1)
    return image

if __name__ == "__main__":
    img2display = gen_sample_4()
    cv2.imshow("Sample Image", img2display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()