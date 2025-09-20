import numpy as np
import cv2

# 方法1 - 用凸性缺陷找凹点
def find_concave_points_defects(contours, epsilon=1.0, depth_threshold=1000):
    """
    用凸性缺陷找凹点
    :param contours: 轮廓列表
    :param epsilon: Douglas-Peucker近似阈值（可选，预简化）
    :param depth_threshold: 缺陷深度阈值，超过才认为是凹点
    :return: (凹点数组, debug_info) where debug_info = {'hulls': list of hull_points arrays, 'defects': list of (point, depth)}
    
    P.S. 一个空白区域只会找到一个凹点，想要更多应该调整epsilon而不是depth_threshold
    """
    concave_points = []
    point_values = []  # 所有缺陷点的(点, 深度)
    hulls = []

    for contour in contours:
        # 预简化轮廓，减少噪声
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        
        if len(approx) < 3:
            continue  # 轮廓太小跳过
        
        hull = cv2.convexHull(approx, returnPoints=False)
        hull_indices = hull.squeeze()
        hull_points = approx[hull_indices].squeeze().astype(np.float32)
        hulls.append(hull_points)
        
        defects = cv2.convexityDefects(approx, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far_point = approx[f][0].astype(np.float32)
                point_values.append((far_point, d))
                # 只有深度超过阈值的才认为是凹点
                if d > depth_threshold:
                    concave_points.append(far_point)
    
    concave_points = np.array(concave_points) if concave_points else np.array([])
    debug_info = {'hulls': hulls, 'defects': point_values}
    return concave_points, debug_info

# 方法2 - 用向量夹角找凹点
def find_concave_points_angle(contours, k=2, angle_threshold=60):
    """
    用局部向量夹角找凹点
    :param contours: 轮廓列表
    :param k: 两侧平均点数
    :param angle_threshold: 夹角阈值（度），小于此值认为是凹点
    :return: (凹点数组, debug_info) where debug_info = {'angles': list of (point, angle)}
    """
    concave_points = []
    point_values = []  # 所有点的(点, 夹角)

    for contour in contours:
        points = contour.squeeze().astype(np.float32)
        n = len(points)
        if n < 2 * k + 1:
            continue  # 轮廓太小跳过

        for i in range(n):
            # 前k个点平均
            prev_indices = (np.arange(-k, 0) + i) % n
            prev_avg = points[prev_indices].mean(axis=0)
            
            # 后k个点平均
            next_indices = (np.arange(1, k + 1) + i) % n
            next_avg = points[next_indices].mean(axis=0)
            
            # 向量
            v1 = points[i] - prev_avg
            v2 = next_avg - points[i]
            
            # 归一化
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                continue
            v1 /= norm_v1
            v2 /= norm_v2
            
            # 点积和夹角
            dot = np.dot(v1, v2)
            dot = np.clip(dot, -1, 1)
            angle_deg = np.degrees(np.arccos(dot))
            
            point_values.append((points[i], angle_deg))
            
            if angle_deg > angle_threshold:
                concave_points.append(points[i])
    
    concave_points = np.array(concave_points) if concave_points else np.array([])
    debug_info = {'angles': point_values}
    return concave_points, debug_info

def remove_too_close_pts(points, image, percentage_threshold=5):
    """
    过滤距离过近的交点
    points: 交点坐标列表
    image: 图像用于计算对角线长度
    percentage_threshold: 距离阈值占图像对角线长度的百分比
    """
    if points.size == 0:  # 使用 .size 检查np数组是否为空
        return []
    
    # 计算图像对角线长度
    height, width = image.shape[:2]
    diagonal_length = np.sqrt(height**2 + width**2)
    
    # 计算最小允许距离（百分比阈值）
    min_distance = diagonal_length * (percentage_threshold / 100)
    
    # 过滤距离过近的点
    filtered_points = []
    
    for i, point1 in enumerate(points):
        keep_point = True
        
        for j, point2 in enumerate(points):
            if i != j:  # 不与自己比较
                # 计算两点间距离
                distance = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
                
                if distance < min_distance:
                    # 如果距离过近，保留先出现的点
                    if i > j:
                        keep_point = False
                        break
        
        if keep_point:
            filtered_points.append(point1)
    
    return filtered_points