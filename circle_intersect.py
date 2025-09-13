"""
基于角度突变检测圆与圆的交点
从重叠圆的轮廓中检测出各个圆的交点位置
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 我的组件
from get_contours import get_outer_inner_contours
from sample_pts import uniform_sampling
from concave_pts import remove_too_close_pts, detect_intersection_points_angle

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 封装：分别处理各个内、外轮廓
def process_contours(contours, image, sampling_interval, angle_threshold, percentage_threshold, contour_type):
    """
    处理给定的轮廓列表（外或内），检测交点并生成图像
    contours: 轮廓列表
    image: 原图像
    sampling_interval: 采样间隔
    angle_threshold: 角度阈值
    percentage_threshold: 稳定化百分比阈值
    contour_type: 'outer' 或 'inner' 用于打印信息
    返回: (intersection_points_list, heatmap_img)
    """
    intersection_points_list = []
    heatmap_img = image.copy()
    
    for i, contour in enumerate(contours):
        print(f"处理{contour_type}轮廓 {i+1}/{len(contours)}")
        
        # 均匀采样
        uniform_sampled = uniform_sampling(contour, sampling_interval)
        
        # 检测交点
        intersection_points, angle_values = detect_intersection_points_angle(uniform_sampled, angle_threshold)
        
        # 稳定化交点
        stabilized_points = remove_too_close_pts(intersection_points, image, percentage_threshold)
        
        # 临时打印被接纳交点的角度分布
        if stabilized_points:
            print(f"{contour_type}轮廓 {i+1} 被接纳交点的角度分布:")
            for point in stabilized_points:
                # 找到该点在采样点中的索引
                points_flat = uniform_sampled.reshape(-1, 2)
                distances = np.sqrt((points_flat[:, 0] - point[0])**2 + 
                                    (points_flat[:, 1] - point[1])**2)
                closest_idx = np.argmin(distances)
                angle_value = angle_values[closest_idx]
                print(f"  交点 ({point[0]}, {point[1]}): 角度值 = {angle_value:.3f} 弧度")
        
        intersection_points_list.append(stabilized_points)
        
        points_flat = uniform_sampled.reshape(-1, 2)
        for point in points_flat:
            cv2.circle(heatmap_img, tuple(map(int, point)), 1, (255, 0, 0), -1)  # 蓝色采样点
        
        # 绘制交点
        for point in stabilized_points:
            cv2.circle(heatmap_img, tuple(map(int, point)), 8, (0, 255, 255), -1)  # 黄色交点
    
    return intersection_points_list, heatmap_img

def main():
    # 读取图像
    image = cv2.imread('imgs/1_half_half.jpg')
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

    # 处理外轮廓
    outer_intersection_points_list, outer_heatmap_img = process_contours(
        outer_contours, image, sampling_interval=4, angle_threshold=0.5, 
        percentage_threshold=1, contour_type="外"
    )

    # 处理内轮廓
    inner_intersection_points_list, inner_heatmap_img = process_contours(
        inner_contours, image, sampling_interval=5, angle_threshold=0.5, 
        percentage_threshold=3, contour_type="内"
    )

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
    axs[1].set_title('外轮廓采样点和交点')
    axs[1].axis('off')

    # 第三列：内轮廓处理结果
    axs[2].imshow(inner_heatmap_img_rgb)
    axs[2].set_title('内轮廓采样点和交点')
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