import cv2

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
