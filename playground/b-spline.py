"""
1、在白色背景上生成三个黑色实心圆形，大小不一
确保三个圆形相交，中间留有空洞。
1.5、预处理（灰度、高斯、二值化）
2、调用cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)找到轮廓
3、调用from get_contours import get_outer_inner_contours获取内外轮廓
4、B-样条splprep对各条轮廓分别进行全局拟合，测试大s
5、分别展示一行二列：所有轮廓点、大s曲线
6、用窗口夹角来找到交点，并筛除掉相聚太近的交点
7、根据交点分割样条，得到真实的圆弧，从而拟合出大部分被遮蔽的圆形

淘汰的内容：
1、小s样条
2、方法一：凸包
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

# === 下面测试找凹点的方法 ===

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

# --- 最小二乘法拟合圆 ---

def fit_circle_least_squares(points):
    """最小二乘拟合圆: 返回 (center_x, center_y, radius)"""
    x, y = points[:, 0], points[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x**2 + y**2)
    try:
        params = np.linalg.lstsq(A, b, rcond=None)[0]  # D, E, F
        cx = -params[0] / 2
        cy = -params[1] / 2
        r = np.sqrt(cx**2 + cy**2 - params[2])
        return cx, cy, r
    except:
        return None  # 拟合失败

# 创建新的可视化窗口
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes3:
    ax.set_aspect('equal')
    ax.imshow(gray, cmap='gray', alpha=0.3)
    ax.invert_yaxis()

axes3[0].set_title('样条分割情况')
axes3[1].set_title('拟合的圆')

colors = ['r', 'g', 'b', 'c', 'm', 'y']
fitted_circles = []

for idx, (points, tck, u) in enumerate(contour_splines):
    color = colors[idx % len(colors)]
    
    # --- 找到凹点对应的u ---
    concave_us = []
    u_dense = np.linspace(0, 1, 100)
    spline_points = splev(u_dense, tck)  # [x_dense, y_dense]
    for concave_pt in concave_pts_angle_filtered:
        dists = np.linalg.norm(spline_points - concave_pt[:, None], axis=0)
        closest_idx = np.argmin(dists)
        concave_us.append(u_dense[closest_idx])
    concave_us = sorted(concave_us)
    
    # --- 分割成段 ---
    segments = []
    u_splits = [0] + concave_us + [1]
    for i in range(len(u_splits) - 1):
        u_seg = np.linspace(u_splits[i], u_splits[i+1], 50)
        seg_points = np.array(splev(u_seg, tck)).T  # (N, 2)
        segments.append(seg_points)
        
        # 绘制分割的段
        axes3[0].plot(seg_points[:, 0], seg_points[:, 1], color=color, linewidth=2, alpha=0.7)
    
    # --- 对每个段拟合圆 ---
    for seg_idx, seg_points in enumerate(segments):
        if len(seg_points) < 3: continue  # 太短跳过
        circle_params = fit_circle_least_squares(seg_points)
        if circle_params is not None:
            cx, cy, r = circle_params
            fitted_circles.append((cx, cy, r))
            
            # 绘制拟合的完整圆
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = cx + r * np.cos(theta)
            circle_y = cy + r * np.sin(theta)
            axes3[1].plot(circle_x, circle_y, '--', color=color, alpha=0.6, linewidth=1.5)
            axes3[1].plot(cx, cy, 'o', color=color, markersize=5)

# 在分割图上标记凹点
if concave_pts_angle_filtered.size > 0:
    axes3[0].scatter(concave_pts_angle_filtered[:, 0], concave_pts_angle_filtered[:, 1], 
                    c='red', s=30, marker='x', linewidths=2, label='凹点')
    axes3[0].legend()

axes3[0].axis('off')
axes3[1].axis('off')
plt.tight_layout()
plt.show()