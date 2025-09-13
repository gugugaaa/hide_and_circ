import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad


# 根据凹点分割样条，并计算每个段的采样点和弧长
def segment_by_concave_points(points, tck, concave_pts_filtered, num_samples=50, min_points=5):
    """
    根据已过滤的凹点分割样条曲线。
    
    参数:
    - points: 轮廓原始点 (np.array, shape=(N, 2))
    - tck: 样条拟合结果
    - concave_pts_filtered: 已过滤的凹点 (np.array, shape=(M, 2))
    - num_samples: 每段采样点数
    - min_points: 最小段长阈值
    
    返回:
    - segments: 列表，每个元素是元组 (seg_points, arc_length)
    """
    # 筛选属于当前轮廓的凹点
    contour_concave_pts = []
    if concave_pts_filtered.size > 0:
        for concave_pt in concave_pts_filtered:
            # 计算凹点到当前轮廓的最小距离
            dists = np.linalg.norm(points - concave_pt, axis=1)
            min_dist = np.min(dists)
            if min_dist < 5:  # 距离阈值，可调整
                contour_concave_pts.append(concave_pt)
    
    # 将凹点投影到样条参数上
    concave_us = []
    u_dense = np.linspace(0, 1, 200)  
    spline_points = np.array(splev(u_dense, tck)).T
    for concave_pt in contour_concave_pts:
        dists = np.linalg.norm(spline_points - concave_pt, axis=1)
        min_dist = np.min(dists)
        if min_dist < 2:
            closest_idx = np.argmin(dists)
            concave_us.append(u_dense[closest_idx])
    concave_us = sorted(set(concave_us))  # 去重并排序
    
    # 分割u区间
    u_splits = [0] + concave_us + [1]
    segments = []
    
    for i in range(len(u_splits) - 1):
        u_start = u_splits[i]
        u_end = u_splits[i + 1]
        
        # 计算弧长
        def integrand(u):
            dx_du, dy_du = splev(u, tck, der=1)
            return np.sqrt(dx_du**2 + dy_du**2)
        arc_length = quad(integrand, u_start, u_end)[0]
        
        # 采样点
        u_seg = np.linspace(u_start, u_end, num_samples)
        seg_points = np.array(splev(u_seg, tck)).T
        
        # 只保留足够长的段
        if len(seg_points) >= min_points:
            segments.append((seg_points, arc_length))
    
    return segments