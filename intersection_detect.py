import numpy as np

def detect_intersection_points_angle(sampled_points, angle_threshold=1.0):
    """
    使用方向突变（turning angle）检测交点
    sampled_points: 均匀采样后的点集，形状 (n, 1, 2)
    angle_threshold: 角度阈值（弧度），默认1.0
    返回: 
        intersection_points: 交点坐标列表
        angles: 每个点的角度值数组（用于调试）
    """
    # 展平点集
    points_flat = sampled_points.reshape(-1, 2)
    n = len(points_flat)
    
    # 计算每个点的 turning angle
    angles = []
    for i in range(n):
        # 获取前一个点、当前点、下一个点，处理闭合轮廓
        prev = points_flat[(i - 1) % n]
        curr = points_flat[i]
        next_ = points_flat[(i + 1) % n]
        
        # 计算向量
        vec_in = curr - prev
        vec_out = next_ - curr
        
        # 使用点积计算夹角（余弦值）
        dot_product = np.dot(vec_in, vec_out)
        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)
        
        # 避免除以零
        if norm_in > 0 and norm_out > 0:
            cos_angle = dot_product / (norm_in * norm_out)
            # 将余弦值转换为角度（弧度），使用arccos
            angle_diff = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        else:
            angle_diff = 0
        
        angles.append(angle_diff)
    
    angles = np.array(angles)
    
    # 找到角度大于阈值的点
    sharp_indices = np.where(angles > angle_threshold)[0]
    intersection_points = [tuple(points_flat[i]) for i in sharp_indices]
    
    return intersection_points, angles

def stabilize_intersection_points(points, image, percentage_threshold=5):
    """
    过滤距离过近的交点
    points: 交点坐标列表
    image: 图像用于计算对角线长度
    percentage_threshold: 距离阈值占图像对角线长度的百分比
    """
    if not points:
        return points
    
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