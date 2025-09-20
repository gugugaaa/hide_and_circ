import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad


# 根据凹点分割样条，并计算每个段的采样点和弧长
def segment_by_concave_points(points, tck, concave_pts_filtered, num_samples=50, min_points=5, min_arc_length=5):
    """
    根据已过滤的凹点分割样条曲线（处理周期性闭合曲线）。
    
    参数:
    - points: 轮廓原始点 (np.array, shape=(N, 2))
    - tck: 样条拟合结果（必须使用 per=1 拟合）
    - concave_pts_filtered: 已过滤的凹点 (np.array, shape=(M, 2))
    - num_samples: 每段采样点数
    - min_points: 最小点数阈值
    - min_arc_length: 新增：最小弧长阈值，用于过滤小段
    
    返回:
    - segments: 列表，每个元素是元组 (seg_points, arc_length)
    """
    # 筛选属于当前轮廓的凹点
    contour_concave_pts = []
    if concave_pts_filtered.size > 0:
        for concave_pt in concave_pts_filtered:
            dists = np.linalg.norm(points - concave_pt, axis=1)
            min_dist = np.min(dists)
            if min_dist < 10:
                contour_concave_pts.append(concave_pt)
    
    # 将凹点投影到样条参数上
    concave_us = []
    u_dense = np.linspace(0, 1, 200)
    spline_points = np.array(splev(u_dense, tck)).T
    for concave_pt in contour_concave_pts:
        dists = np.linalg.norm(spline_points - concave_pt, axis=1)
        min_dist = np.min(dists)
        if min_dist < 5:
            closest_idx = np.argmin(dists)
            concave_us.append(u_dense[closest_idx])
    concave_us = sorted(set(concave_us))  # 去重并排序
    
    # 如果没有凹点，整个曲线作为一个段
    if not concave_us:
        def integrand(u):
            dx_du, dy_du = splev(u, tck, der=1)
            return np.sqrt(dx_du**2 + dy_du**2)
        arc_length = quad(integrand, 0, 1)[0]
        u_seg = np.linspace(0, 1, num_samples)
        seg_points = np.array(splev(u_seg, tck)).T
        if len(seg_points) >= min_points and arc_length >= min_arc_length:
            return [(seg_points, arc_length)]
        return []
    
    # 分割u区间（周期性处理）
    segments = []
    for i in range(len(concave_us)):
        u_start = concave_us[i]
        u_end = concave_us[(i + 1) % len(concave_us)]
        wrap = u_end <= u_start
        if wrap:
            u_end += 1
        
        # 计算弧长（如果wrap，分割积分）
        def integrand(u):
            dx_du, dy_du = splev(u % 1, tck, der=1)  # %1 以确保在[0,1]
            return np.sqrt(dx_du**2 + dy_du**2)
        if wrap:
            arc_length = quad(integrand, u_start, 1)[0] + quad(integrand, 0, u_end - 1)[0]
        else:
            arc_length = quad(integrand, u_start, u_end)[0]
        
        # 采样点（如果wrap，分割采样并连接）
        if wrap:
            # 按长度比例分配采样点
            len1 = 1 - u_start
            len2 = u_end - 1
            total_len = len1 + len2
            num1 = int(num_samples * (len1 / total_len)) + 1  # +1确保连接
            num2 = num_samples - num1 + 1  # 调整重叠
            u_seg1 = np.linspace(u_start, 1, num1)
            u_seg2 = np.linspace(0, u_end - 1, num2)
            seg_points1 = np.array(splev(u_seg1 % 1, tck)).T
            seg_points2 = np.array(splev(u_seg2 % 1, tck)).T
            seg_points = np.vstack((seg_points1[:-1], seg_points2))  # 去掉重复的端点
        else:
            u_seg = np.linspace(u_start, u_end, num_samples)
            seg_points = np.array(splev(u_seg % 1, tck)).T
        
        # 只保留足够长的段
        if len(seg_points) >= min_points and arc_length >= min_arc_length:
            segments.append((seg_points, arc_length))
    
    # 打印调试信息
    for concave_pt in contour_concave_pts:
        dists = np.linalg.norm(points - concave_pt, axis=1)
        min_dist = np.min(dists)
        print(f"Concave pt {np.round(concave_pt, 1)}: min_dist to points = {min_dist:.1f}")
    for concave_pt in contour_concave_pts:
        dists = np.linalg.norm(spline_points - concave_pt, axis=1)
        min_dist = np.min(dists)
        print(f"Concave pt {np.round(concave_pt, 1)}: min_dist to spline_points = {min_dist:.1f}")
    print(f"Filtered contour_concave_pts: {[np.round(pt, 1) for pt in contour_concave_pts]}")
    print(f"Projected concave_us: {[round(u, 1) for u in concave_us]}")
    
    return segments