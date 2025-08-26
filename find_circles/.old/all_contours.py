"""
轮廓调试工具 - 绘制所有轮廓及其序号
用于调试和分析轮廓检测结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def debug_draw_all_contours(image_path):
    """
    调试函数：绘制所有轮廓及其序号
    使用与intersection_pts.py相同的轮廓检测方法
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化处理
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓 - 使用与intersection_pts.py相同的参数
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(f"找到 {len(contours)} 个轮廓")

    # 创建用于绘制轮廓的图像副本
    contour_image = image.copy()
    
    # 生成不同颜色用于区分轮廓
    colors = []
    for i in range(len(contours)):
        # 使用HSV色彩空间生成不同颜色
        hue = int(180 * i / len(contours))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])))

    # 绘制所有轮廓
    for i, contour in enumerate(contours):
        # 绘制轮廓
        cv2.drawContours(contour_image, [contour], -1, colors[i], 2)
        
        # 计算轮廓的质心用于放置序号
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 如果质心计算失败，使用边界矩形的中心
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

        # 在质心位置绘制序号
        cv2.putText(contour_image, str(i), (cx, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(contour_image, str(i), (cx, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # 转换BGR到RGB以便plt显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # 创建1行3列的子图布局
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 第一列：原图
    axs[0].imshow(image_rgb)
    axs[0].set_title('原图')
    axs[0].axis('off')

    # 第二列：二值化图像
    axs[1].imshow(binary_rgb)
    axs[1].set_title('二值化图像')
    axs[1].axis('off')

    # 第三列：轮廓图像
    axs[2].imshow(contour_image_rgb)
    axs[2].set_title(f'所有轮廓 (共{len(contours)}个)')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印轮廓信息
    print("\n轮廓详细信息:")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # 层次结构信息
        if hierarchy is not None:
            next_idx, prev_idx, child_idx, parent_idx = hierarchy[0][i]
            hierarchy_info = f"next:{next_idx}, prev:{prev_idx}, child:{child_idx}, parent:{parent_idx}"
        else:
            hierarchy_info = "无层次信息"
        
        print(f"轮廓 {i}: 面积={area:.1f}, 周长={perimeter:.1f}, "
              f"边界框=({x},{y},{w},{h}), 层次=({hierarchy_info})")

def main():
    # 调试指定图像
    debug_draw_all_contours('imgs/9.jpg')

if __name__ == "__main__":
    main()
