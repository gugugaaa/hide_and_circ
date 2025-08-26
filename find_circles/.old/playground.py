"""
最终目的：从重叠的圆的黑箱中，猜测出各个圆

1、加载图片，拿到find contours点集，绘制出来

2、处理近似点集

3、找到不可导点（两圆弧交点）  当前尝试中……
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  # 新增导入
from scipy.ndimage import gaussian_filter  # 新增导入

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取图像
image = cv2.imread('imgs/7.jpg')
if image is None:
    print("Error: Could not load image")
    exit()

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 二值化处理
_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# 新增函数：筛选外轮廓（排除图像边框）
def get_outer_contour(contours, image_shape):
    """
    从轮廓列表中筛选出重叠圆的外轮廓，排除图像边框轮廓
    contours: 轮廓列表
    image_shape: 图像形状 (height, width)
    """
    height, width = image_shape[:2]
    outer_contours = []
    
    for contour in contours:
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 排除图像边框轮廓（轮廓与图像边界接触）
        margin = 5  # 边界容差
        if (x > margin and y > margin and 
            x + w < width - margin and 
            y + h < height - margin):
            outer_contours.append(contour)
    
    # 按面积降序排序，返回最大的轮廓（最可能是外轮廓）
    if outer_contours:
        outer_contours = sorted(outer_contours, key=cv2.contourArea, reverse=True)
        return outer_contours[0]
    else:
        # 如果没有找到合适的轮廓，返回第一个轮廓（备选方案）
        return contours[0]

# 查找轮廓 - 使用 CHAIN_APPROX_NONE 获取所有点
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 改为LIST模式
print(f"找到 {len(contours)} 个轮廓")

# 获取外轮廓
outer_contour = get_outer_contour(contours, image.shape)
print(f"找到 {len(contours)} 个轮廓，使用外轮廓进行后续处理")

# 等弧长均匀采样函数
def uniform_sampling(contour, num_points):
    """
    沿着轮廓进行等弧长均匀采样
    contour: 轮廓点集
    num_points: 需要采样的点数
    """
    # 计算轮廓总长度
    total_length = 0
    for i in range(len(contour) - 1):
        pt1 = contour[i][0]
        pt2 = contour[i + 1][0]
        total_length += np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    
    # 闭合轮廓，添加最后一个点到第一个点的距离
    pt1 = contour[-1][0]
    pt2 = contour[0][0]
    total_length += np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    
    # 计算采样间隔
    interval = total_length / num_points
    
    sampled_points = []
    current_length = 0
    accumulated_length = 0
    
    # 遍历轮廓点进行采样
    for i in range(len(contour)):
        # 计算当前段的长度
        if i < len(contour) - 1:
            pt1 = contour[i][0]
            pt2 = contour[i + 1][0]
            segment_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        else:
            # 最后一段：最后一个点到第一个点
            pt1 = contour[-1][0]
            pt2 = contour[0][0]
            segment_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        
        # 在当前段中采样
        while accumulated_length + segment_length >= current_length + interval:
            # 计算采样点的位置
            t = (current_length + interval - accumulated_length) / segment_length
            if i < len(contour) - 1:
                x = pt1[0] + t * (pt2[0] - pt1[0])
                y = pt1[1] + t * (pt2[1] - pt1[1])
            else:
                x = pt1[0] + t * (pt2[0] - pt1[0])
                y = pt1[1] + t * (pt2[1] - pt1[1])
            
            sampled_points.append([[int(x), int(y)]])
            current_length += interval
            
            # 如果已经采样足够多的点，极速退出
            if len(sampled_points) >= num_points:
                break
        
        accumulated_length += segment_length
        if len(sampled_points) >= num_points:
            break
    
    return np.array(sampled_points)

# 使用等弧长均匀采样
num_sample_points = 400  # 可以根据需要调整采样点数
uniform_sampled = uniform_sampling(outer_contour, num_sample_points)  # 修改为使用outer_contour

# 将原函数重命名为detect_intersection_points_hull
def detect_intersection_points_hull(contour, distance_threshold=500):
    """
    使用凸缺陷检测方法找到轮廓中的交点
    contour: 输入轮廓
    distance_threshold: 距离阈值，用于过滤显著的缺陷点
    """
    # 计算凸包（返回索引）
    hull = cv2.convexHull(contour, returnPoints=False)
    
    # 计算凸缺陷
    defects = cv2.convexityDefects(contour, hull)
    
    intersection_points = []
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]  # 起始点、结束点、最远点索引、距离
            if d > distance_threshold:  # 距离阈值，d是放大10000倍的值
                far_point = tuple(contour[f][0])
                intersection_points.append(far_point)
    
    return intersection_points, hull

# 使用多尺度曲率检测方法替换原来的曲率检测方法
def detect_intersection_points_curvature(sampled_points, percentile_threshold=95):
    """
    使用多尺度曲率检测方法找到轮廓中的交点
    sampled_points: 均匀采样后的点集
    percentile_threshold: 曲率峰值检测的百分位阈值 (0-100)
    """
    # 提取x和y坐标
    points_flat = sampled_points.reshape(-1, 2)
    x = points_flat[:, 0]
    y = points_flat[:, 1]
    
    # 初始化曲率数组
    curvature_combined = np.zeros(len(x))
    
    # 多尺度窗口计算曲率 (3-9步长2)
    for window_size in range(3, 10, 2):
        # 计算导数（使用循环差分处理闭合轮廓）
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        ddx = np.diff(dx, append=dx[0])
        ddy = np.diff(dy, append=dy[0])
        
        # 计算曲率
        numerator = np.abs(dx[:-1] * ddy[:-1] - dy[:-1] * ddx[:-1])
        denominator = (dx[:-1]**2 + dy[:-1]**2 + 1e-8)**1.5
        curvature = numerator / denominator
        
        # 使用高斯滤波平滑曲率
        curvature_smoothed = gaussian_filter(curvature, sigma=window_size/2)
        
        # 将当前尺度的曲率添加到组合曲率中
        curvature_combined[:-1] += curvature_smoothed
        curvature_combined[-1] += curvature_smoothed[0]  # 处理闭合轮廓
    
    # 归一化曲率值到[0,1]范围
    if np.max(curvature_combined) > 0:
        curvature_normalized = curvature_combined / np.max(curvature_combined)
    else:
        curvature_normalized = np.zeros_like(curvature_combined)
    
    # 使用动态百分位阈值
    height_threshold = np.percentile(curvature_normalized, percentile_threshold)
    
    # 检测曲率峰值（交点）
    peaks, _ = find_peaks(curvature_normalized, height=height_threshold)
    
    # 获取交点坐标
    intersection_points = [tuple(points_flat[i]) for i in peaks]
    
    # 使用matplotlib的colormap进行颜色映射
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    colormap = cm.get_cmap('jet')  # 可选：'plasma', 'viridis', 'turbo'等
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # 创建热力图图像
    heatmap_img = np.ones_like(image) * 255

    # 绘制曲率热图
    for i, point in enumerate(points_flat):
        color_float = colormap(norm(curvature_normalized[i]))  # RGBA, 0-1
        color_bgr = (
            int(color_float[2] * 255),  # B
            int(color_float[1] * 255),  # G
            int(color_float[0] * 255),  # R
        )
        cv2.circle(heatmap_img, tuple(point), 2, color_bgr, -1)
    
    return intersection_points, curvature_normalized, heatmap_img

# 新增：基于方向突变（turning angle）检测交点的函数
def detect_intersection_points_angle(sampled_points, angle_threshold=1.0):
    """
    使用方向突变（turning angle）检测交点
    sampled_points: 均匀采样后的点集，形状 (n, 1, 2)
    angle_threshold: 角度阈值（弧度），默认1.0
    返回: 
        intersection_points: 交点坐标列表
        angles: 每个点的角度值数组（用于调试）
        heatmap_img: 角度热力图
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
        
        # 计算角度差
        angle_in = np.arctan2(vec_in[1], vec_in[0])
        angle_out = np.arctan2(vec_out[1], vec_out[0])
        angle_diff = angle_out - angle_in
        
        # 归一化到 [-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        angles.append(angle_diff)
    
    angles = np.array(angles)
    
    # 找到角度绝对值大于阈值的点
    sharp_indices = np.where(np.abs(angles) > angle_threshold)[0]
    intersection_points = [tuple(points_flat[i]) for i in sharp_indices]
    
    # 创建角度热力图
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # 使用绝对值作为热力图的值
    angle_abs = np.abs(angles)
    if np.max(angle_abs) > 0:
        angle_normalized = angle_abs / np.max(angle_abs)
    else:
        angle_normalized = np.zeros_like(angle_abs)
    
    colormap = cm.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # 创建热力图图像
    heatmap_img = np.ones_like(image) * 255
    
    # 绘制角度热图
    for i, point in enumerate(points_flat):
        color_float = colormap(norm(angle_normalized[i]))  # RGBA, 0-1
        color_bgr = (
            int(color_float[2] * 255),  # B
            int(color_float[1] * 255),  # G
            int(color_float[0] * 255),  # R
        )
        cv2.circle(heatmap_img, tuple(point), 2, color_bgr, -1)
    
    return intersection_points, angles, heatmap_img

# 新增：稳定器函数，过滤距离过近的交点
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

# 检测交点（使用三种方法）
intersection_points_hull, hull_indices = detect_intersection_points_hull(outer_contour, 300)  # 修改为使用outer_contour
intersection_points_curvature, curvature_values, curvature_heatmap_img = detect_intersection_points_curvature(uniform_sampled, 90)
intersection_points_angle, angle_values, angle_heatmap_img = detect_intersection_points_angle(uniform_sampled, 0.5)

# 应用稳定器过滤距离过近的交点
stabilized_hull = stabilize_intersection_points(intersection_points_hull, image, 5)
stabilized_curvature = stabilize_intersection_points(intersection_points_curvature, image, 5)
stabilized_angle = stabilize_intersection_points(intersection_points_angle, image, 5)

# 在热力图上绘制稳定后的交点（黄色大点）
for point in stabilized_curvature:
    cv2.circle(curvature_heatmap_img, point, 8, (50, 150, 230), -1)  # 黄色大点表示交点

for point in stabilized_angle:
    cv2.circle(angle_heatmap_img, point, 8, (50, 150, 230), -1)  # 黄色大点表示交点

# 创建白色背景图像
white_bg = np.ones_like(image) * 255
uniform_pts = white_bg.copy()
hull_img = white_bg.copy()

# 绘制均匀采样点
for point in uniform_sampled:
    x, y = point[0]
    cv2.circle(uniform_pts, (x, y), 1, (0, 0, 255), -1)  # 红色点

# 绘制凸包和交点（凸缺陷方法）
hull_points = cv2.convexHull(outer_contour, returnPoints=True)  # 修改为使用outer_contour
cv2.drawContours(hull_img, [hull_points], -1, (0, 255, 0), 2)
for point in intersection_points_hull:
    cv2.circle(hull_img, point, 5, (255, 0, 0), -1)  # 蓝色点

# 转换BGR到RGB以便plt显示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
uniform_pts = cv2.cvtColor(uniform_pts, cv2.COLOR_BGR2RGB)
hull_img = cv2.cvtColor(hull_img, cv2.COLOR_BGR2RGB)
curvature_heatmap_img = cv2.cvtColor(curvature_heatmap_img, cv2.COLOR_BGR2RGB)
angle_heatmap_img = cv2.cvtColor(angle_heatmap_img, cv2.COLOR_BGR2RGB)  # 新增

# 修改：创建3x2的子图布局，隐藏第三行第二列
fig, axs = plt.subplots(3, 2)

# 第一行
axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title('原图')
axs[0, 0].axis('off')

axs[0, 1].imshow(uniform_pts)
axs[0, 1].set_title('均匀采样点集')
axs[0, 1].axis('off')

# 第二行
axs[1, 0].imshow(hull_img)
axs[1, 0].set_title('凸包(绿色)和交点(蓝色)-凸缺陷方法')
axs[1, 0].axis('off')

axs[1, 1].imshow(curvature_heatmap_img)
axs[1, 1].set_title('多尺度曲率热图和交点(黄色大点)')
axs[1, 1].axis('off')

# 第三行
axs[2, 0].imshow(angle_heatmap_img)
axs[2, 0].set_title('角度突变热图和交点(黄色大点)')
axs[2, 0].axis('off')

# 隐藏第三行第二列
axs[2, 1].axis('off')

plt.tight_layout()
plt.show()

# 打印检测到的交点信息
print(f"凸缺陷方法原本检测到 {len(intersection_points_hull)} 个交点，稳定后 {len(stabilized_hull)} 个交点:")
for i, point in enumerate(stabilized_hull):
    print(f"交点 {i+1}: 坐标 ({point[0]}, {point[1]})")

print(f"\n曲率方法原本检测到 {len(intersection_points_curvature)} 个交点，稳定后 {len(stabilized_curvature)} 个交点:")
for i, point in enumerate(stabilized_curvature):
    print(f"交点 {i+1}: 坐标 ({point[0]}, {point[1]})")

print(f"\n角度方法原本检测到 {len(intersection_points_angle)} 个交点，稳定后 {len(stabilized_angle)} 个交点:")
for i, point in enumerate(stabilized_angle):
    print(f"交点 {i+1}: 坐标 ({point[0]}, {point[1]})")