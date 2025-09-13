import numpy as np
from collections import defaultdict

from . import ArcFitResult

def calculate_circle_iou(circle1, circle2):
    """
    计算两个圆的IOU（基于面积重叠）。
    circle1/2: ArcFitResult实例，有center_x, center_y, radius。
    """
    d = np.sqrt((circle1.center_x - circle2.center_x)**2 + (circle1.center_y - circle2.center_y)**2)
    r1, r2 = circle1.radius, circle2.radius
    if d >= r1 + r2:  # 无交集
        return 0.0
    if d <= abs(r1 - r2):  # 一个包含另一个
        return min(np.pi * r1**2, np.pi * r2**2) / max(np.pi * r1**2, np.pi * r2**2)
    # 部分重叠
    d1 = (r1**2 - r2**2 + d**2) / (2 * d)
    d2 = d - d1
    area1 = r1**2 * np.arccos(d1 / r1) - d1 * np.sqrt(r1**2 - d1**2)
    area2 = r2**2 * np.arccos(d2 / r2) - d2 * np.sqrt(r2**2 - d2**2)
    intersection = area1 + area2
    union = np.pi * r1**2 + np.pi * r2**2 - intersection
    return intersection / union

def merge_candidates_by_iou(filter_results, iou_threshold=0.9, min_group_size=2):
    """
    对filter_clusters_by_arc_length的结果进行进一步的IOU合并。
    对于仍有多个候选的簇，基于pairwise IOU合并。
    
    :param filter_results: filter_clusters_by_arc_length的输出结果 (list of (ArcFitResult or list of ArcFitResult))
    :param iou_threshold: IOU阈值，默认0.9
    :param min_group_size: 最小组大小，默认2
    :return: results (list)，每个元素对应一个簇的最佳圆 (ArcFitResult) 或整个cluster
    """
    results = []
    for result in filter_results:
        if result is None:
            continue  # 跳过无效簇
        elif not isinstance(result, list):
            # 已是单个最优圆，直接保留
            results.append(result)
            continue
        
        # 仍有多个候选的簇
        candidates = result  # list of ArcFitResult
        if len(candidates) < 2:
            # 不需要合并，直接返回
            if len(candidates) == 1:
                results.append(candidates[0])
            continue
        
        # 计算所有pairwise IOU
        iou_matrix = np.zeros((len(candidates), len(candidates)))
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                iou = calculate_circle_iou(candidates[i], candidates[j])
                iou_matrix[i, j] = iou_matrix[j, i] = iou
        
        # 找出高IOU组
        # 这里用简单方式：找所有pairwise >= threshold的连接组件（graph）
        graph = defaultdict(list)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if iou_matrix[i, j] >= iou_threshold:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # 找最大连通组（多数派）
        visited = set()
        max_group = []
        for node in range(len(candidates)):
            if node not in visited:
                group = dfs(graph, node, visited)  # 用DFS找连通组件
                if len(group) >= min_group_size and len(group) > len(max_group):
                    max_group = group
        
        if not max_group:
            # 无高IOU组，不满足条件时原封不动返回整个cluster
            results.append(candidates)
            continue
        
        # 平均最大组的圆（加权平均）
        group_circles = [candidates[i] for i in max_group]
        weights = [c.arc_length for c in group_circles]  # 加权用arc_length
        total_weight = sum(weights)
        avg_x = sum(c.center_x * w for c, w in zip(group_circles, weights)) / total_weight
        avg_y = sum(c.center_y * w for c, w in zip(group_circles, weights)) / total_weight
        avg_r = sum(c.radius * w for c, w in zip(group_circles, weights)) / total_weight
        avg_arc = max(c.arc_length for c in group_circles)  # 或平均，但max更保守
        
        merged_circle = ArcFitResult(avg_x, avg_y, avg_r, avg_arc)
        results.append(merged_circle)
    
    return results

# 辅助DFS找连通组件
def dfs(graph, node, visited):
    stack = [node]
    component = []
    while stack:
        curr = stack.pop()
        if curr not in visited:
            visited.add(curr)
            component.append(curr)
            stack.extend(graph[curr])
    return component