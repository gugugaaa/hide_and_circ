"""
1、在白色背景上生成三个黑色实心圆形，大小不一
确保三个圆形相交，中间留有空洞。
1.5、预处理（灰度、高斯、二值化）
2、调用cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)找到轮廓
3、调用from get_contours import get_outer_inner_contours获取内外轮廓
4、B-样条splprep对各条轮廓分别进行全局拟合，测试大s和小s
5、分别展示一行三列：所有轮廓点、大s曲线、小s曲线

PS：
1、由于光栅化和锯齿效应，对角线处的黑色台阶更密集，会更容易正确拟合。而最上和最下容易横平竖直，拟合我直线
所以我在想，这一步先把凹点（圆弧的交点）找到，之后再分段拟合圆弧。

2、测试发现小s或者采样点数太高没有并不会再提升描述能力。
"""

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 我的 playground 组件
from get_contours import get_outer_inner_contours

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 1. 生成三个相交的圆形
image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# 绘制三个相交的圆
cv2.circle(image, (150, 150), 80, (0, 0, 0), -1)
cv2.circle(image, (250, 150), 60, (0, 0, 0), -1)
cv2.circle(image, (200, 250), 70, (0, 0, 0), -1)

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
for contour in outer_contours + inner_contours:
    points = contour.squeeze()
    tck, u = splprep(points.T, s=s_large, k=3)  # 大s值，平滑
    u_new = np.linspace(0, 1, 200)
    x_new, y_new = splev(u_new, tck)
    axes[1].plot(x_new, y_new, 'b-')
axes[1].set_title(f'大s平滑拟合 (s={s_large})')
axes[1].invert_yaxis()

plt.show()

# === 下面测试找突变点的方法 ===
