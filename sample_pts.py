import numpy as np

def uniform_sampling(contour, interval):
    """
    沿着轮廓进行等弧长均匀采样
    contour: 轮廓点集
    interval: 采样点间距（像素）
    """
    contour = contour.astype(np.float32)  # 确保浮点
    total_length = 0.0
    for i in range(len(contour) - 1):
        pt1 = contour[i][0]
        pt2 = contour[i + 1][0]
        total_length += np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    pt1 = contour[-1][0]
    pt2 = contour[0][0]
    total_length += np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    
    num_points = int(total_length / interval) + 1  # +1 确保闭合
    
    sampled_points = []
    current_length = 0.0
    accumulated_length = 0.0
    
    for i in range(len(contour)):
        if i < len(contour) - 1:
            pt1 = contour[i][0]
            pt2 = contour[i + 1][0]
            segment_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        else:
            pt1 = contour[-1][0]
            pt2 = contour[0][0]
            segment_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        
        while accumulated_length + segment_length >= current_length + interval and len(sampled_points) < num_points:
            remaining = current_length + interval - accumulated_length
            t = remaining / segment_length if segment_length > 0 else 0.0
            x = pt1[0] + t * (pt2[0] - pt1[0])
            y = pt1[1] + t * (pt2[1] - pt1[1])
            sampled_points.append([x, y])
            current_length += interval
        
        accumulated_length += segment_length
    
    # 闭合：添加第一个点
    if sampled_points:
        sampled_points.append(sampled_points[0])
    
    return np.array(sampled_points)
