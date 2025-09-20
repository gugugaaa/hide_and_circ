import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# 我的组件
import utils.gen_sample as gen
from utils.get_contours import get_outer_inner_contours
from utils.concave_pts import find_concave_points_angle, remove_too_close_pts
from utils.seg_spline import segment_by_concave_points
from utils.fit_circle import fit_arc_to_circle
from utils.circle_cluster import filter_circles_by_iou, cluster_circles, filter_clusters_by_arc_length
from utils.merge_by_iou import merge_candidates_by_iou

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# === 步骤1、画图 ===
image = gen.gen_sample_2()

# === 步骤1.5、灰度高斯二值化 ===
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

# === 步骤2、获取内外轮廓 ===
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
outer_contours, inner_contours = get_outer_inner_contours(contours, hierarchy, image.shape[:2])

# === 步骤3、B样条拟合 s=20 各条轮廓，采样200个点 ===
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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

# B样条拟合 (s=20)
s = 20
contour_splines = []  # 保存每个轮廓的样条拟合结果
for contour in outer_contours + inner_contours:
    points = contour.squeeze()
    tck, u = splprep(points.T, s=s, k=3, per=1)
    contour_splines.append((points, tck, u))
    u_new = np.linspace(0, 1, 200)
    x_new, y_new = splev(u_new, tck)
    axes[1].plot(x_new, y_new, 'b-')
axes[1].set_title(f'B样条拟合 (s={s})')
axes[1].invert_yaxis()

# === 步骤4、找圆弧交点（凹点） ===
concave_pts, debug_angle = find_concave_points_angle(outer_contours + inner_contours, k=4, angle_threshold=40)
concave_pts_filtered = np.array(remove_too_close_pts(concave_pts, gray, percentage_threshold=2))

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
axes2[0].set_title('夹角法-夹角分布')
axes2[0].imshow(gray, cmap='gray')
if debug_angle['angles']:
    angle_points = np.array([pt for pt, angle in debug_angle['angles']])
    angles = np.array([angle for pt, angle in debug_angle['angles']])
    if angle_points.size > 0:
        scatter2 = axes2[0].scatter(angle_points[:, 0], angle_points[:, 1], c=angles, cmap='jet', s=10)
        plt.colorbar(scatter2, ax=axes2[0], label='角度')
axes2[0].invert_yaxis()
axes2[0].axis('off')

axes2[1].set_title('夹角法-凹点分布')
axes2[1].imshow(gray, cmap='gray')
if concave_pts_filtered.size > 0:
    axes2[1].scatter(concave_pts_filtered[:, 0], concave_pts_filtered[:, 1], c='r', s=15)
axes2[1].invert_yaxis()
axes2[1].axis('off')
plt.tight_layout()

# === 步骤5、切割样条，采样得到新圆弧 ===
# 创建新的可视化窗口
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))
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
    
    # 分割样条
    segments = segment_by_concave_points(points, tck, concave_pts_filtered, num_samples=50, min_points=5)
    
    # 可视化分割段
    for seg_points, arc_length in segments:
        axes3[0].plot(seg_points[:, 0], seg_points[:, 1], color=color, linewidth=2, alpha=0.7)
    
    # === 步骤6、对采样得到的圆弧分别拟合圆 ===
    for seg_points, arc_length in segments:
        fitted_circle = fit_arc_to_circle(seg_points, arc_length)
        if fitted_circle is not None:
            fitted_circles.append(fitted_circle)
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = fitted_circle.center_x + fitted_circle.radius * np.cos(theta)
            circle_y = fitted_circle.center_y + fitted_circle.radius * np.sin(theta)
            axes3[1].plot(circle_x, circle_y, '--', color=color, alpha=0.6, linewidth=1.5)
            axes3[1].plot(fitted_circle.center_x, fitted_circle.center_y, 'o', color=color, markersize=5)

# 在分割图上标记凹点
if concave_pts_filtered.size > 0:
    axes3[0].scatter(concave_pts_filtered[:, 0], concave_pts_filtered[:, 1], 
                     c='blue', s=30, marker='x', linewidths=2, label='凹点')
    axes3[0].legend()

axes3[0].axis('off')
axes3[1].axis('off')
plt.tight_layout()

# === 步骤7、对拟合圆分簇和初筛 ===
# 步骤7.1、清除拟合圆脱离黑色实心区域的
filtered_circles = filter_circles_by_iou(fitted_circles, binary, min_iou=0.9, verbose=True)

# 步骤7.2、分簇
clusters = cluster_circles(filtered_circles, threshold=0.2)

# 步骤7.3、采纳可信的拟合圆：弧长够大的
filter_results = filter_clusters_by_arc_length(clusters, min_arc_ratio=0.5, arc_diff_threshold=1.5)

# === 步骤8、合并还有候选的簇：根据簇内IOU ===
final_results = merge_candidates_by_iou(filter_results, iou_threshold=0.9, min_group_size=2)

# 扁平化最终圆列表
clustered_circles = []
for result in final_results:
    if isinstance(result, list):
        clustered_circles.extend(result)
    else:
        clustered_circles.append(result)

if clustered_circles:
    # 创建新的可视化窗口
    fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))
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
    for i, result in enumerate(final_results):
        color = cluster_colors[i % len(cluster_colors)]
        
        if isinstance(result, list):
            for circle in result:
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = circle.center_x + circle.radius * np.cos(theta)
                circle_y = circle.center_y + circle.radius * np.sin(theta)
                axes4[1].plot(circle_x, circle_y, '--', color=color, alpha=0.6, linewidth=1.5)
                axes4[1].plot(circle.center_x, circle.center_y, 'o', color=color, markersize=5)
        else:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = result.center_x + result.radius * np.cos(theta)
            circle_y = result.center_y + result.radius * np.sin(theta)
            axes4[1].plot(circle_x, circle_y, '-', color=color, alpha=0.8, linewidth=2)
            axes4[1].plot(result.center_x, result.center_y, 'o', color=color, markersize=6)
    
    plt.tight_layout()

plt.show()