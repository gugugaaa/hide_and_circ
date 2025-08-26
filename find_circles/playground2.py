"""
基于角度突变检测圆与圆的交点
从重叠圆的轮廓中检测出各个圆的交点

尝试：
先用默认为3的窗口对直出的密集点集平滑，
然后再进行uniform sampling。

新增smooth point(contour, window=3)函数并使用

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_outer_inner_contours(contours: list, hierarchy: np.ndarray | None, image_shape: tuple[int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def uniform_sampling(contour: np.ndarray, interval: float) -> np.ndarray:
    """
    沿着轮廓进行等弧长均匀采样
    contour: 轮廓点集，形状 (n, 1, 2)
    interval: 采样点间距（像素）
    返回: 采样后的点集，形状 (m, 1, 2)
    """
    # 展平点集
    points = contour.reshape(-1, 2)
    n = len(points)
    
    # 计算轮廓总长度（闭合）
    total_length = 0
    for i in range(n):
        pt1 = points[i]
        pt2 = points[(i + 1) % n]
        total_length += np.linalg.norm(pt2 - pt1)
    
    # 计算采样点数
    num_points = int(total_length / interval) + 1  # 确保至少一个点
    
    sampled_points = []
    current_length = 0
    accumulated_length = 0
    i = 0
    
    while len(sampled_points) < num_points and i < n * 2:  # 防止无限循环
        pt1 = points[i % n]
        pt2 = points[(i + 1) % n]
        segment_length = np.linalg.norm(pt2 - pt1)
        
        while accumulated_length + segment_length >= current_length + interval:
            remaining = current_length + interval - accumulated_length
            t = remaining / segment_length
            x = pt1[0] + t * (pt2[0] - pt1[0])
            y = pt1[1] + t * (pt2[1] - pt1[1])
            sampled_points.append([[int(x), int(y)]])
            current_length += interval
            if len(sampled_points) >= num_points:
                break
        
        accumulated_length += segment_length
        i += 1
    
    return np.array(sampled_points)


def detect_intersection_points_angle(sampled_points: np.ndarray, angle_threshold: float = 1.0) -> tuple[list[tuple[int, int]], np.ndarray]:
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
    angles = np.zeros(n)
    for i in range(n):
        prev = points_flat[(i - 1) % n]
        curr = points_flat[i]
        next_ = points_flat[(i + 1) % n]
        
        vec_in = curr - prev
        vec_out = next_ - curr
        
        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)
        
        if norm_in > 0 and norm_out > 0:
            cos_angle = np.dot(vec_in, vec_out) / (norm_in * norm_out)
            angles[i] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # 找到角度大于阈值的点
    sharp_indices = np.where(angles > angle_threshold)[0]
    intersection_points = [tuple(points_flat[i]) for i in sharp_indices]
    
    return intersection_points, angles


def stabilize_intersection_points(points: list[tuple[int, int]], image_shape: tuple[int, int], percentage_threshold: float = 5) -> list[tuple[int, int]]:
    """
    过滤距离过近的交点
    points: 交点坐标列表
    image_shape: 图像形状 (height, width)
    percentage_threshold: 距离阈值占图像对角线长度的百分比
    返回: 过滤后的交点列表
    """
    if not points:
        return points
    
    height, width = image_shape[:2]
    diagonal_length = np.sqrt(height**2 + width**2)
    min_distance = diagonal_length * (percentage_threshold / 100)
    
    filtered_points = []
    for i, point1 in enumerate(points):
        keep = True
        for j in range(i):
            point2 = points[j]
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            if distance < min_distance:
                keep = False
                break
        if keep:
            filtered_points.append(point1)
    
    return filtered_points


def smooth_points(contour: np.ndarray, window: int = 3) -> np.ndarray:
    """
    使用滑动窗口平均法平滑轮廓点
    contour: 轮廓点集，形状 (n, 1, 2)
    window: 滑动窗口大小，必须为奇数
    返回: 平滑后的点集，形状 (n, 1, 2)
    """
    if window % 2 == 0:
        window += 1
    
    points = contour.reshape(-1, 2)
    n = len(points)
    smoothed_points = np.zeros_like(points)
    
    half_window = window // 2
    for i in range(n):
        indices = [(i + j) % n for j in range(-half_window, half_window + 1)]
        window_points = points[indices]
        smoothed_points[i] = np.mean(window_points, axis=0).astype(int)
    
    return smoothed_points.reshape(-1, 1, 2)


def process_contour(
    contour: np.ndarray,
    sampling_interval: float,
    angle_threshold: float,
    distance_threshold: float,
    image_shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """
    处理单个轮廓：平滑、采样、检测交点
    contour: 轮廓点集，形状 (n, 1, 2)
    sampling_interval: 采样间隔
    angle_threshold: 角度阈值
    distance_threshold: 距离阈值百分比
    image_shape: 图像形状，用于稳定化交点
    返回: (stabilized_points, uniform_sampled, angle_values)
    """
    smoothed_contour = smooth_points(contour, window=3)
    uniform_sampled = uniform_sampling(smoothed_contour, sampling_interval)
    intersection_points, angle_values = detect_intersection_points_angle(uniform_sampled, angle_threshold)
    stabilized_points = stabilize_intersection_points(intersection_points, image_shape, distance_threshold)
    
    return stabilized_points, uniform_sampled, angle_values


def draw_smoothed_contours(
    image: np.ndarray,
    outer_contours: list[np.ndarray],
    inner_contours: list[np.ndarray]
) -> np.ndarray:
    """
    绘制所有轮廓的平滑点到图像上
    image: 基础图像
    outer_contours: 外轮廓列表
    inner_contours: 内轮廓列表
    返回: 绘制后的图像
    """
    result_img = image.copy()
    # 外轮廓：绿色
    for contour in outer_contours:
        smoothed = smooth_points(contour, window=3)
        points_flat = smoothed.reshape(-1, 2)
        for point in points_flat:
            cv2.circle(result_img, tuple(point.astype(int)), 1, (0, 255, 0), -1)
    # 内轮廓：蓝色
    for contour in inner_contours:
        smoothed = smooth_points(contour, window=3)
        points_flat = smoothed.reshape(-1, 2)
        for point in points_flat:
            cv2.circle(result_img, tuple(point.astype(int)), 1, (255, 0, 0), -1)
    return result_img


def draw_processed_results(
    image: np.ndarray,
    processed_results: list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray]],
    show_heatmap: bool
) -> np.ndarray:
    """
    绘制处理结果到共享图像上（采样点/热力图 + 交点）
    image: 基础图像
    processed_results: 每个轮廓的 (stabilized_points, uniform_sampled, angle_values) 列表
    show_heatmap: 是否绘制热力图（否则绘制蓝色采样点）
    返回: 绘制后的图像
    """
    result_img = image.copy()
    if show_heatmap:
        # 为热力图准备 colormap
        colormap = cm.get_cmap('jet')
        norm = mcolors.Normalize(vmin=0, vmax=1)
    
    for stabilized_points, uniform_sampled, angle_values in processed_results:
        points_flat = uniform_sampled.reshape(-1, 2).astype(int)
        
        if show_heatmap:
            # 归一化角度（绝对值）
            angle_abs = np.abs(angle_values)
            max_angle = np.max(angle_abs) if np.max(angle_abs) > 0 else 1
            angle_normalized = angle_abs / max_angle
            
            for i, point in enumerate(points_flat):
                color_float = colormap(norm(angle_normalized[i]))
                color_bgr = (
                    int(color_float[2] * 255),
                    int(color_float[1] * 255),
                    int(color_float[0] * 255)
                )
                cv2.circle(result_img, tuple(point), 2, color_bgr, -1)
        else:
            # 绘制蓝色采样点
            for point in points_flat:
                cv2.circle(result_img, tuple(point), 1, (255, 0, 0), -1)
        
        # 绘制黄色交点
        for point in stabilized_points:
            cv2.circle(result_img, tuple(point), 8, (0, 255, 255), -1)
    
    return result_img


def print_intersection_details(
    intersection_points_list: list[list[tuple[int, int]]],
    processed_results: list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray]],
    contour_type: str
) -> None:
    """
    打印检测到的交点信息和角度分布
    intersection_points_list: 每个轮廓的交点列表
    processed_results: 每个轮廓的 (stabilized_points, uniform_sampled, angle_values) 列表
    contour_type: "外轮廓" 或 "内轮廓"
    """
    total_points = sum(len(points) for points in intersection_points_list)
    if total_points > 0:
        print(f"{contour_type}总共检测到 {total_points} 个交点:")
        for i, (points, (_, uniform_sampled, angle_values)) in enumerate(zip(intersection_points_list, processed_results)):
            print(f"{contour_type} {i+1}: {len(points)} 个交点")
            points_flat = uniform_sampled.reshape(-1, 2)
            for j, point in enumerate(points):
                distances = np.linalg.norm(points_flat - np.array(point), axis=1)
                closest_idx = np.argmin(distances)
                angle_value = angle_values[closest_idx] if closest_idx < len(angle_values) else None
                print(f"  交点 {j+1}: ({point[0]}, {point[1]})" +
                      (f" - 角度值 = {angle_value:.3f} 弧度" if angle_value is not None else " - 无法获取角度值"))
    else:
        print(f"未检测到{contour_type}交点")


def main() -> None:
    # 设置选项
    SHOW_HEATMAP = False  # 是否显示热力图

    # 读取图像
    image = cv2.imread('imgs/9.jpg')
    if image is None:
        print("Error: Could not load image")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化处理
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(f"找到 {len(contours)} 个轮廓")

    # 获取外轮廓和内轮廓
    outer_contours, inner_contours = get_outer_inner_contours(contours, hierarchy, image.shape)
    print(f"找到 {len(outer_contours)} 个外轮廓，{len(inner_contours)} 个内轮廓")

    # 定义参数
    sampling_interval = 4
    angle_threshold = 0.55
    distance_threshold = 1

    # 处理外轮廓
    outer_processed_results: list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray]] = []
    outer_intersection_points_list: list[list[tuple[int, int]]] = []
    for i, contour in enumerate(outer_contours):
        print(f"处理外轮廓 {i+1}/{len(outer_contours)}")
        stabilized_points, uniform_sampled, angle_values = process_contour(
            contour, sampling_interval, angle_threshold, distance_threshold, image.shape
        )
        outer_processed_results.append((stabilized_points, uniform_sampled, angle_values))
        outer_intersection_points_list.append(stabilized_points)

    # 处理内轮廓
    inner_processed_results: list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray]] = []
    inner_intersection_points_list: list[list[tuple[int, int]]] = []
    for i, contour in enumerate(inner_contours):
        print(f"处理内轮廓 {i+1}/{len(inner_contours)}")
        stabilized_points, uniform_sampled, angle_values = process_contour(
            contour, sampling_interval, angle_threshold, distance_threshold, image.shape
        )
        inner_processed_results.append((stabilized_points, uniform_sampled, angle_values))
        inner_intersection_points_list.append(stabilized_points)

    # 绘制图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    smoothed_contours_img = draw_smoothed_contours(image, outer_contours, inner_contours)
    smoothed_contours_img_rgb = cv2.cvtColor(smoothed_contours_img, cv2.COLOR_BGR2RGB)
    
    outer_result_img = draw_processed_results(image, outer_processed_results, SHOW_HEATMAP)
    outer_result_img_rgb = cv2.cvtColor(outer_result_img, cv2.COLOR_BGR2RGB)
    
    inner_result_img = draw_processed_results(image, inner_processed_results, SHOW_HEATMAP)
    inner_result_img_rgb = cv2.cvtColor(inner_result_img, cv2.COLOR_BGR2RGB)

    # 创建2行2列的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 第一行第一列：原图
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title('原图')
    axs[0, 0].axis('off')

    # 第一行第二列：平滑后的轮廓
    axs[0, 1].imshow(smoothed_contours_img_rgb)
    axs[0, 1].set_title('平滑后的轮廓')
    axs[0, 1].axis('off')

    # 第二行第一列：外轮廓处理结果
    axs[1, 0].imshow(outer_result_img_rgb)
    title_text = f'外轮廓{"热力图" if SHOW_HEATMAP else "采样点"}和交点'
    axs[1, 0].set_title(title_text)
    axs[1, 0].axis('off')

    # 第二行第二列：内轮廓处理结果
    axs[1, 1].imshow(inner_result_img_rgb)
    title_text = f'内轮廓{"热力图" if SHOW_HEATMAP else "采样点"}和交点'
    axs[1, 1].set_title(title_text)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印检测到的交点信息
    print_intersection_details(outer_intersection_points_list, outer_processed_results, "外轮廓")
    print_intersection_details(inner_intersection_points_list, inner_processed_results, "内轮廓")


if __name__ == "__main__":
    main()