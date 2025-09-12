import numpy as np
import cv2

# 方法1 - 凸缺陷检测凹点
def find_concave_points_defects(contours, epsilon=1.0):
    """
    使用凸缺陷找到凹点。
    :param contours: 轮廓列表
    :param epsilon: 简化轮廓的Douglas-Peucker阈值（可选，用于预简化）
    :return: 凹点列表 (numpy array)
    """
    concave_points = []
    for contour in contours:
        # 预简化轮廓以减少噪声
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        hull = cv2.convexHull(approx, returnPoints=False)
        if len(hull) < 3:
            continue  # 跳过太小的轮廓
        
        defects = cv2.convexityDefects(approx, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # d 是缺陷深度，阈值过滤小缺陷（e.g., d > 1000）
                if d > 1000:  # 调整此阈值基于图像大小
                    far_point = approx[f][0]  # 最远点作为凹点
                    concave_points.append(far_point)
    
    return np.array(concave_points) if concave_points else np.array([])
