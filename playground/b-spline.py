"""
1、在白色背景上生成多个黑色实心圆形，大小不一，例如三个圆形相交，中间留有空洞。
1.5、预处理（灰度、高斯、二值化）
2、调用cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)找到轮廓
3、调用get_outer_inner_contours获取内外轮廓
4、B-样条splprep对各条轮廓分别进行全局拟合，测试大s
5、分别展示一行二列：所有轮廓点、大s曲线
6、用窗口夹角来找到交点，并筛除掉相聚太近的交点
7、根据交点分割样条，得到真实的圆弧，从而拟合出大部分被遮蔽的圆形
7.5：清除掉某些圆和有效区域（黑色实心）IOU小于90%的，来避免小圆弧错误拟合到白色背景侵入的圆
8、对拟合出来的圆形分割为簇
9、对每个簇，如果有多个候选且arc length相差很大，并且最大弧本身足够大，那么选用最长弧的拟合结果
10、对于仍有多个候选的簇，如果多个候选圆的IOU相差不大，那么使用综合平均的结果

淘汰的内容：
1、小s样条——无意义
2、方法一：凸包——太慢
3、用聚类分组——太慢
"""

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 我的组件
import gen_sample as gen
from get_contours import get_outer_inner_contours
from concave_pts import find_concave_points_angle, remove_too_close_pts
from seg_spline import segment_by_concave_points
from fit_circle import fit_arc_to_circle
from circle_cluster import cluster_circles, filter_clusters_by_arc_length, calculate_iou_with_black, filter_circles_by_iou

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 1. 生成图像
image = gen.gen_sample_3()

# 1.5 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

# 2. 找到轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 3. 获取内外轮廓
outer_contours, inner_contours = get_outer_inner_contours(contours, hierarchy, image.shape)

# 4. B样条拟合
fig, axes = plt.subplots(1, 2)
for ax in axes:
    ax.set_aspect('equal') 

# 原始轮廓点
all_points = []
for contour in outer_contours + inner_contours:
    points = contour.squeeze()
    all_points.extend(points)
    axes[0].plot(points[:, 0], points[:, 1], 'o', markersize=2)
axes[0].set_title('原始轮廓点')
axes[0].invert_yaxis()

# 大s平滑拟合 (s=s_large)
s_large = 20
contour_splines = []  # 保存每个轮廓的样条拟合结果
for contour in outer_contours + inner_contours:
    points = contour.squeeze()
    tck, u = splprep(points.T, s=s_large, k=3)  # 大s值，平滑
    contour_splines.append((points, tck, u))
    u_new = np.linspace(0, 1, 200)
    x_new, y_new = splev(u_new, tck)
    axes[1].plot(x_new, y_new, 'b-')
axes[1].set_title(f'大s平滑拟合 (s={s_large})')
axes[1].invert_yaxis()

# plt.show()

# === 测试找凹点的方法 ===

# 方法2：夹角法
concave_pts_angle, debug_angle = find_concave_points_angle(outer_contours + inner_contours, k=4, angle_threshold=50)
# 过滤凹点
concave_pts_angle_filtered = np.array(remove_too_close_pts(concave_pts_angle.tolist(), gray, percentage_threshold=5))

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
axes2[0].set_title('夹角法-夹角分布')
axes2[0].imshow(gray, cmap='gray')
if debug_angle['angles']:
    angle_points = np.array([pt for pt, c in debug_angle['angles']])
    angle_concavities = np.array([c for pt, c in debug_angle['angles']])
    if angle_points.size > 0:
        scatter2 = axes2[0].scatter(angle_points[:, 0], angle_points[:, 1], c=angle_concavities, cmap='jet', s=10)
        plt.colorbar(scatter2, ax=axes2[0], label='凹凸性角度')
axes2[0].invert_yaxis()
axes2[0].axis('off')

axes2[1].set_title('夹角法-凹点分布')
axes2[1].imshow(gray, cmap='gray')
if concave_pts_angle_filtered.size > 0:
    axes2[1].scatter(concave_pts_angle_filtered[:, 0], concave_pts_angle_filtered[:, 1], c='r', s=15)
axes2[1].invert_yaxis()
axes2[1].axis('off')
plt.tight_layout()
# plt.show()

# === 根据凹点预测圆弧 ===

# 创建新的可视化窗口
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes3:
    ax.set_aspect('equal')
    ax.imshow(gray, cmap='gray', alpha=0.3)
    ax.invert_yaxis()

axes3[0].set_title('样条分割情况')
axes3[1].set_title('拟合的圆')

fitted_circles = []
colors = ['r', 'g', 'b', 'c', 'm', 'y']

for idx, (points, tck, u) in enumerate(contour_splines):
    color = colors[idx % len(colors)]
    
    # 使用seg_spline函数进行分割
    segments_with_arc = segment_by_concave_points(points, tck, concave_pts_angle_filtered)
    
    # 可视化分割段
    for seg_points, arc_length in segments_with_arc:
        axes3[0].plot(seg_points[:, 0], seg_points[:, 1], color=color, linewidth=2, alpha=0.7)
    
    # 拟合圆
    for seg_points, arc_length in segments_with_arc:
        fitted_circle = fit_arc_to_circle(seg_points, arc_length)
        """
        fitted_circle是一个ArcFitResult类，包括center_x, center_y, radius, arc_length
        """
        if fitted_circle is not None:
            fitted_circles.append(fitted_circle)
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = fitted_circle.center_x + fitted_circle.radius * np.cos(theta)
            circle_y = fitted_circle.center_y + fitted_circle.radius * np.sin(theta)
            axes3[1].plot(circle_x, circle_y, '--', color=color, alpha=0.6, linewidth=1.5)
            axes3[1].plot(fitted_circle.center_x, fitted_circle.center_y, 'o', color=color, markersize=5)

# 在分割图上标记凹点
if concave_pts_angle_filtered.size > 0:
    axes3[0].scatter(concave_pts_angle_filtered[:, 0], concave_pts_angle_filtered[:, 1], 
                    c='red', s=30, marker='x', linewidths=2, label='凹点')
    axes3[0].legend()

axes3[0].axis('off')
axes3[1].axis('off')
plt.tight_layout()
# plt.show()

# === 步骤7.5: IOU过滤 ===

# 使用封装的函数过滤fitted_circles
filtered_circles = filter_circles_by_iou(fitted_circles, binary, min_iou=0.9, verbose=True)

# === 把圆分割为簇 ===

# 先分簇 - 使用过滤后的圆
clusters = cluster_circles(filtered_circles, threshold=0.2)

# 再过滤每个簇
filter_results = filter_clusters_by_arc_length(clusters, min_arc_ratio=0.5, arc_diff_threshold=2.0)

# clustered_circles: 过滤后的最佳圆列表（排除None）
clustered_circles = [r for r in filter_results if r is not None]

if clustered_circles:
    # 创建新的可视化窗口
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes4:
        ax.set_aspect('equal')
        ax.imshow(gray, cmap='gray', alpha=0.3)
        ax.invert_yaxis()
        ax.axis('off')
    
    axes4[0].set_title('基于偏差指标分组后的拟合圆')
    axes4[1].set_title('弧长过滤后的最佳圆')
    
    # 颜色列表
    cluster_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # 为每个簇绘制圆（第一列）
    for i, cluster in enumerate(clusters):
        color = cluster_colors[i % len(cluster_colors)]
        
        for circle in cluster:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = circle.center_x + circle.radius * np.cos(theta)
            circle_y = circle.center_y + circle.radius * np.sin(theta)
            axes4[0].plot(circle_x, circle_y, '--', color=color, alpha=0.6, linewidth=1.5)
            axes4[0].plot(circle.center_x, circle.center_y, 'o', color=color, markersize=5)
    
    # 绘制过滤后的最佳圆（第二列）
    for i, best_circle in enumerate(filter_results):
        if best_circle is not None:
            color = cluster_colors[i % len(cluster_colors)]
            # 显示最佳圆
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = best_circle.center_x + best_circle.radius * np.cos(theta)
            circle_y = best_circle.center_y + best_circle.radius * np.sin(theta)
            axes4[1].plot(circle_x, circle_y, '-', color=color, alpha=0.8, linewidth=2)
            axes4[1].plot(best_circle.center_x, best_circle.center_y, 'o', color=color, markersize=6)
        else:
            # 过滤条件不满足，显示该簇的所有候选圆形
            color = cluster_colors[i % len(cluster_colors)]
            for circle in clusters[i]:
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = circle.center_x + circle.radius * np.cos(theta)
                circle_y = circle.center_y + circle.radius * np.sin(theta)
                axes4[1].plot(circle_x, circle_y, '--', color=color, alpha=0.6, linewidth=1.5)
                axes4[1].plot(circle.center_x, circle.center_y, 'o', color=color, markersize=5)
    
    plt.tight_layout()

plt.show()