# circle_cluster.py

import numpy as np
import cv2

def calculate_iou_with_black(circle, binary_image):
    # 创建圆掩码（与图像同大小）
    mask = np.zeros_like(binary_image)
    cv2.circle(mask, (int(circle.center_x), int(circle.center_y)), int(circle.radius), 255, -1)
    
    # 交集：mask AND binary (binary中黑色=255)
    intersection = cv2.bitwise_and(mask, binary_image)
    intersection_area = np.count_nonzero(intersection)  # 交集像素数
    
    # 圆面积（像素近似）
    circle_area = np.count_nonzero(mask)  # 或用πr²，但像素计数更准
    
    iou = intersection_area / circle_area if circle_area > 0 else 0
    return iou


def relative_deviation_score(circle1, circle2):
    """
    计算两个圆的相对偏差分数。
    :param circle1: ArcFitResult 对象
    :param circle2: ArcFitResult 对象
    :return: 相对偏差分数 (float)
    """
    cx1, cy1, r1 = circle1.center_x, circle1.center_y, circle1.radius
    cx2, cy2, r2 = circle2.center_x, circle2.center_y, circle2.radius
    d = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)  # 中心距离
    dr = abs(r1 - r2)  # 半径差异
    avg_r = (r1 + r2) / 2  # 平均半径作为动态幅值
    if avg_r == 0:  # 避免除零
        return float('inf')
    score = (d / avg_r) + (dr / avg_r)  # 相对偏差指标
    return score

def cluster_circles(fitted_circles, threshold=0.2):
    """
    使用贪心合并算法将拟合的圆形分成簇。
    :param fitted_circles: 列表，包含 ArcFitResult 对象
    :param threshold: 偏差阈值，小于此视为同一组
    :return: clusters (list of lists)，每个子列表是一个簇的圆形列表
    """
    if not fitted_circles:
        return []
    
    # 初始化每个圆作为一个簇
    clusters = [[c] for c in fitted_circles]
    
    # 迭代合并
    merged = True
    while merged:
        merged = False
        new_clusters = []
        while clusters:
            current = clusters.pop(0)
            for i in range(len(clusters) - 1, -1, -1):  # 从后往前检查，避免索引问题
                other = clusters[i]
                # 计算簇平均与另一个簇平均的score
                avg_current = np.mean([[c.center_x, c.center_y, c.radius] for c in current], axis=0)
                avg_other = np.mean([[c.center_x, c.center_y, c.radius] for c in other], axis=0)
                dummy_circle1 = type('Dummy', (), {'center_x': avg_current[0], 'center_y': avg_current[1], 'radius': avg_current[2]})
                dummy_circle2 = type('Dummy', (), {'center_x': avg_other[0], 'center_y': avg_other[1], 'radius': avg_other[2]})
                score = relative_deviation_score(dummy_circle1, dummy_circle2)
                if score < threshold:
                    current.extend(other)  # 合并
                    del clusters[i]
                    merged = True
            new_clusters.append(current)
        clusters = new_clusters
    
    return clusters

def filter_clusters_by_arc_length(clusters, min_arc_ratio=0.5, arc_diff_threshold=2.0):
    """
    对每个簇应用弧长过滤：
    - 如果簇只有一个圆，直接保留。
    - 如果多个候选且 max_arc / min_arc > arc_diff_threshold，并且最大弧 > min_arc_ratio * 2πr，则选用最长弧的拟合结果。
    - 否则，返回整个簇（不满足条件时原封不动返回cluster）。
    :param clusters: list of lists，每个子列表是一个簇的圆形列表
    :param min_arc_ratio: 最大弧需 > min_arc_ratio * 2πr (默认0.5，即50%)
    :param arc_diff_threshold: 如果 max_arc / min_arc > threshold，则视为相差很大
    :return: results (list)，每个元素对应一个簇的最佳圆 (ArcFitResult) 或整个cluster
    """
    results = []
    for cluster in clusters:
        if len(cluster) == 1:
            results.append(cluster[0])  # 只有一个，直接保留
            continue
        
        # 提取每个圆的arc_length和radius
        arc_lengths = [c.arc_length for c in cluster]
        max_arc_idx = np.argmax(arc_lengths)
        max_arc = arc_lengths[max_arc_idx]
        min_arc = min(arc_lengths)
        
        # 检查相差是否很大
        diff_ratio = max_arc / min_arc if min_arc > 0 else float('inf')
        is_large_diff = diff_ratio > arc_diff_threshold
        
        # 检查最大弧是否足够大
        max_circle = cluster[max_arc_idx]
        circumference = 2 * np.pi * max_circle.radius
        is_large_enough = max_arc > (min_arc_ratio * circumference)
        
        if is_large_diff and is_large_enough:
            results.append(max_circle)  # 选择最长弧的拟合结果
        else:
            results.append(cluster)  # 不满足条件时返回整个cluster
            
    return results

def filter_circles_by_iou(fitted_circles, binary_image, min_iou=0.9, verbose=False):
    """
    根据与黑色区域的IOU过滤圆形。
    :param fitted_circles: 列表，包含 ArcFitResult 对象
    :param binary_image: 二值图像（黑色=255）
    :param min_iou: IOU阈值，默认0.9
    :param verbose: 是否打印调试信息
    :return: 过滤后的圆形列表
    """
    filtered_circles = []
    for circle in fitted_circles:
        iou = calculate_iou_with_black(circle, binary_image)
        if iou >= min_iou:
            filtered_circles.append(circle)
        elif verbose:
            print(f"剔除圆: center=({circle.center_x:.1f}, {circle.center_y:.1f}), radius={circle.radius:.1f}, IOU={iou:.2f}")
    
    if verbose:
        print(f"IOU过滤前: {len(fitted_circles)}个圆，过滤后: {len(filtered_circles)}个圆")
    
    return filtered_circles