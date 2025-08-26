"""
基于角度突变检测圆与圆的交点
从重叠圆的轮廓中检测出各个圆的交点位置
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_outer_inner_contours(contours, hierarchy, image_shape):
    """
    从轮廓列表中筛选出重叠圆的外轮廓和内轮廓，排除图像边框轮廓
    contours: 轮廓列表
    hierarchy: 轮廓层次结构
    image_shape: 图像形状 (height, width)
    返回: (outer_contours, inner_contours) - 两个列表
    """
    height, width = image_shape[:2]
    
    outer_contours = []
    inner_contours = []
    
    if hierarchy is None:
        # 如果没有层次结构，将所有有效轮廓视为外轮廓
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            margin = 5
            if (x > margin and y > margin and 
                x + w < width - margin and 
                y + h < height - margin):
                outer_contours.append(contour)
        return outer_contours, inner_contours
    
    # 筛选有效的轮廓（不接触图像边界）
    valid_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        margin = 5
        if (x > margin and y > margin and 
            x + w < width - margin and 
            y + h < height - margin):
            valid_contours.append((i, contour))
    
    if not valid_contours:
        return outer_contours, inner_contours
    
    # 根据层次结构分类轮廓
    for idx, contour in valid_contours:
        parent_idx = hierarchy[0][idx][3]  # parent
        
        # 如果没有父轮廓或父轮廓是图像边界，则为外轮廓
        if parent_idx == -1:
            outer_contours.append(contour)
        else:
            # 检查父轮廓是否也是有效轮廓
            parent_contour = contours[parent_idx]
            x, y, w, h = cv2.boundingRect(parent_contour)
            margin = 5
            if (x > margin and y > margin and 
                x + w < width - margin and 
                y + h < height - margin):
                inner_contours.append(contour)
            else:
                # 如果父轮廓无效（可能是边界），则当前轮廓为外轮廓
                outer_contours.append(contour)
    
    return outer_contours, inner_contours

def uniform_sampling(contour, interval):
    """
    沿着轮廓进行等弧长均匀采样
    contour: 轮廓点集
    interval: 采样点间距（像素）
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
    
    # 计算采样点数
    num_points = int(total_length / interval)
    
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
            
        accumulated_length += segment_length
        if len(sampled_points) >= num_points:
            break
    
    return np.array(sampled_points)

def detect_intersection_points_angle(sampled_points, angle_threshold=1.0):
    """
    使用方向突变（turning angle）检测交点
    sampled_points: 均匀采样后的点集，形状 (n, 1, 2)
    angle_threshold: 角度阈值（弧度），默认1.0
    返回: 
        intersection_points: 交点坐标列表
        angles: 每个点的角度值数组（用于调试）
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
        
        # 使用点积计算夹角（余弦值）
        dot_product = np.dot(vec_in, vec_out)
        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)
        
        # 避免除以零
        if norm_in > 0 and norm_out > 0:
            cos_angle = dot_product / (norm_in * norm_out)
            # 将余弦值转换为角度（弧度），使用arccos
            angle_diff = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        else:
            angle_diff = 0
        
        angles.append(angle_diff)
    
    angles = np.array(angles)
    
    # 找到角度大于阈值的点
    sharp_indices = np.where(angles > angle_threshold)[0]
    intersection_points = [tuple(points_flat[i]) for i in sharp_indices]
    
    return intersection_points, angles

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

def create_angle_heatmap(sampled_points, angles, image):
    """
    创建角度热力图
    """
    points_flat = sampled_points.reshape(-1, 2)
    
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
    
    return heatmap_img

def main():
    # 设置选项
    SHOW_HEATMAP = True  # 是否显示热力图，默认关闭
    
    # 读取图像
    image = cv2.imread('imgs/1_half_half_half.jpg')
    if image is None:
        print("Error: Could not load image")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化处理
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓 - 使用 CHAIN_APPROX_NONE 获取所有点，RETR_TREE 获取层次结构
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(f"找到 {len(contours)} 个轮廓")

    # 获取外轮廓和内轮廓
    outer_contours, inner_contours = get_outer_inner_contours(contours, hierarchy, image.shape)
    print(f"找到 {len(outer_contours)} 个外轮廓，{len(inner_contours)} 个内轮廓")

    # 处理所有外轮廓
    outer_intersection_points_list = []
    outer_heatmap_img = image.copy()
    
    sampling_outer = 4  # 像素间隔
    for i, contour in enumerate(outer_contours):
        print(f"处理外轮廓 {i+1}/{len(outer_contours)}")
        
        # 均匀采样
        uniform_sampled = uniform_sampling(contour, sampling_outer)
        
        # 检测交点
        intersection_points, angle_values = detect_intersection_points_angle(uniform_sampled, 0.5)
        
        # 稳定化交点
        stabilized_points = stabilize_intersection_points(intersection_points, image, 1)
        
        # 临时打印被接纳交点的角度分布
        if stabilized_points:
            print(f"外轮廓 {i+1} 被接纳交点的角度分布:")
            for point in stabilized_points:
                # 找到该点在采样点中的索引
                points_flat = uniform_sampled.reshape(-1, 2)
                distances = np.sqrt((points_flat[:, 0] - point[0])**2 + 
                                  (points_flat[:, 1] - point[1])**2)
                closest_idx = np.argmin(distances)
                angle_value = angle_values[closest_idx]
                print(f"  交点 ({point[0]}, {point[1]}): 角度值 = {angle_value:.3f} 弧度")
        
        outer_intersection_points_list.append(stabilized_points)
        
        # 如果启用热力图，创建并叠加
        if SHOW_HEATMAP:
            heatmap = create_angle_heatmap(uniform_sampled, angle_values, image)
            outer_heatmap_img = cv2.addWeighted(outer_heatmap_img, 0.7, heatmap, 0.3, 0)
        else:
            # 绘制采样点
            points_flat = uniform_sampled.reshape(-1, 2)
            for point in points_flat:
                cv2.circle(outer_heatmap_img, tuple(point), 1, (255, 0, 0), -1)  # 蓝色采样点
        
        # 绘制交点
        for point in stabilized_points:
            cv2.circle(outer_heatmap_img, point, 8, (0, 255, 255), -1)  # 黄色交点

    # 处理所有内轮廓
    inner_intersection_points_list = []
    inner_heatmap_img = image.copy()
    
    for i, contour in enumerate(inner_contours):
        print(f"处理内轮廓 {i+1}/{len(inner_contours)}")
        
        # 均匀采样
        sampling_inner = 5
        uniform_sampled = uniform_sampling(contour, sampling_inner)
        
        # 检测交点
        intersection_points, angle_values = detect_intersection_points_angle(uniform_sampled, 0.5)
        
        # 稳定化交点
        stabilized_points = stabilize_intersection_points(intersection_points, image, 3)
        
        # 临时打印被接纳交点的角度分布
        if stabilized_points:
            print(f"内轮廓 {i+1} 被接纳交点的角度分布:")
            for point in stabilized_points:
                # 找到该点在采样点中的索引
                points_flat = uniform_sampled.reshape(-1, 2)
                distances = np.sqrt((points_flat[:, 0] - point[0])**2 + 
                                  (points_flat[:, 1] - point[1])**2)
                closest_idx = np.argmin(distances)
                angle_value = angle_values[closest_idx]
                print(f"  交点 ({point[0]}, {point[1]}): 角度值 = {angle_value:.3f} 弧度")
        
        inner_intersection_points_list.append(stabilized_points)
        
        # 如果启用热力图，创建并叠加
        if SHOW_HEATMAP:
            heatmap = create_angle_heatmap(uniform_sampled, angle_values, image)
            inner_heatmap_img = cv2.addWeighted(inner_heatmap_img, 0.7, heatmap, 0.3, 0)
        else:
            # 绘制采样点
            points_flat = uniform_sampled.reshape(-1, 2)
            for point in points_flat:
                cv2.circle(inner_heatmap_img, tuple(point), 1, (255, 0, 0), -1)  # 蓝色采样点
        
        # 绘制交点
        for point in stabilized_points:
            cv2.circle(inner_heatmap_img, point, 8, (0, 255, 255), -1)  # 黄色交点

    # 转换BGR到RGB以便plt显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outer_heatmap_img_rgb = cv2.cvtColor(outer_heatmap_img, cv2.COLOR_BGR2RGB)
    inner_heatmap_img_rgb = cv2.cvtColor(inner_heatmap_img, cv2.COLOR_BGR2RGB)

    # 创建1行3列的子图布局
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 第一列：原图
    axs[0].imshow(image_rgb)
    axs[0].set_title('原图')
    axs[0].axis('off')

    # 第二列：外轮廓处理结果
    axs[1].imshow(outer_heatmap_img_rgb)
    title_text = f'外轮廓{"热力图" if SHOW_HEATMAP else "采样点"}和交点'
    axs[1].set_title(title_text)
    axs[1].axis('off')

    # 第三列：内轮廓处理结果
    axs[2].imshow(inner_heatmap_img_rgb)
    title_text = f'内轮廓{"热力图" if SHOW_HEATMAP else "采样点"}和交点'
    axs[2].set_title(title_text)
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印检测到的交点信息
    total_outer_points = sum(len(points) for points in outer_intersection_points_list)
    print(f"外轮廓总共检测到 {total_outer_points} 个交点:")
    for i, points in enumerate(outer_intersection_points_list):
        print(f"外轮廓 {i+1}: {len(points)} 个交点")
        for j, point in enumerate(points):
            print(f"  交点 {j+1}: ({point[0]}, {point[1]})")
    
    total_inner_points = sum(len(points) for points in inner_intersection_points_list)
    if total_inner_points > 0:
        print(f"内轮廓总共检测到 {total_inner_points} 个交点:")
        for i, points in enumerate(inner_intersection_points_list):
            print(f"内轮廓 {i+1}: {len(points)} 个交点")
            for j, point in enumerate(points):
                print(f"  交点 {j+1}: ({point[0]}, {point[1]})")
    else:
        print("未检测到内轮廓交点")

if __name__ == "__main__":
    main()