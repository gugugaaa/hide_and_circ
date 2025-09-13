# 代码接口和各个步骤

```python
# === 步骤1、画图 ===
# --- gen sample.py ---
def gen_sample_3():
    """
    在白色背景生成四个相交的黑色实心圆形，圆心在矩形的四个角
    """

# === 步骤1.5、灰度高斯二值化 ===

# === 步骤2、获取内外轮廓 ===
# --- get contours.py ---
def get_outer_inner_contours(contours, hierarchy, image_shape):
    """
    从轮廓列表中筛选出重叠圆的外轮廓和内轮廓，排除图像边框轮廓
    contours: 轮廓列表
    hierarchy: 轮廓层次结构
    image_shape: 图像形状 (height, width)
    返回: (outer_contours, inner_contours) - 两个列表
    """

# === 步骤3、B样条拟合s=20各条轮廓，采样200个点 ===

# === 步骤4、找圆弧交点（凹点） ===
# --- concave points.py ---
def find_concave_points_angle(contours, k=2, angle_threshold=50):
    """
    用局部向量夹角找凹点
    :param contours: 轮廓列表
    :param k: 两侧平均点数
    :param angle_threshold: 夹角阈值（度），小于此值认为是凹点
    :return: (凹点数组, debug_info) where debug_info = {'angles': list of (point, angle)}
    """

def remove_too_close_pts(points, image, percentage_threshold=5):
    """
    过滤距离过近的交点
    points: 交点坐标列表
    image: 图像用于计算对角线长度
    percentage_threshold: 距离阈值占图像对角线长度的百分比
    """

# === 步骤5、切割样条，采样得到新圆弧 ===
# --- seg spline.py ---
def segment_by_concave_points(points, tck, concave_pts_filtered, num_samples=50, min_points=5):
    """
    根据已过滤的凹点分割样条曲线。
    
    参数:
    - points: 轮廓原始点 (np.array, shape=(N, 2))
    - tck: 样条拟合结果
    - concave_pts_filtered: 已过滤的凹点 (np.array, shape=(M, 2))
    - num_samples: 每段采样点数
    - min_points: 最小段长阈值
    
    返回:
    - segments: 列表，每个元素是元组 (seg_points, arc_length)
    """

# === 步骤6、对采样得到的圆弧分别拟合圆 ===
# --- fit circle.py ---
class ArcFitResult:
    """
    拟合圆弧的结果类
    包含圆心坐标(center_x, center_y), 半径(radius), 以及弧长(arc_length)
    """

def fit_arc_to_circle(seg_points, arc_length):
    """
    根据样条的采样点，拟合圆弧
    返回： ArcFitResult 或 None
    """

# === 步骤7、对拟合圆分簇和初筛
# --- circle cluster.py ---
# --- 步骤7.1、清除拟合圆脱离黑色实心区域的 ---
def filter_circles_by_iou(fitted_circles, binary_image, min_iou=0.9, verbose=False):
    """
    根据与黑色区域的IOU过滤圆形。
    :param fitted_circles: 列表，包含 ArcFitResult 对象
    :param binary_image: 二值图像（黑色=255）
    :param min_iou: IOU阈值，默认0.9
    :param verbose: 是否打印调试信息
    :return: 过滤后的圆形列表
    """
# --- 步骤7.2、分簇 ---
def cluster_circles(fitted_circles, threshold=0.2):
    """
    使用贪心合并算法将拟合的圆形分成簇。
    :param fitted_circles: 列表，包含 ArcFitResult 对象
    :param threshold: 偏差阈值，小于此视为同一组
    :return: clusters (list of lists)，每个子列表是一个簇的圆形列表
    """
# --- 步骤7.3、采纳可信的拟合圆：弧长够大的 ---
def filter_clusters_by_arc_length(clusters, min_arc_ratio=0.5, arc_diff_threshold=1.5):
    """
    对每个簇应用弧长过滤：
    - 如果簇只有一个圆，直接保留。
    - 如果多个候选且 max_arc / min_arc > arc_diff_threshold，并且最大弧 > min_arc_ratio * 2πr，则选用最长弧的拟合结果。
    - 否则，返回整个簇（不满足条件时原封不动返回cluster）。
    :param clusters: list of lists，每个子列表是一个簇的圆形列表
    :param min_arc_ratio: 最大弧需 > min_arc_ratio * 2πr (默认0.5，即50%)
    :param arc_diff_threshold: 如果 max_arc / min_arc > threshold，则视为相差很大
    :return: results (list)，每个元素对应一个簇的最佳圆 (ArcFitResult) 或整个cluster
    """

# === 步骤8、合并还有候选的簇：根据簇内IOU ===
# --- merge_by_iou.py ---
def merge_candidates_by_iou(filter_results, iou_threshold=0.9, min_group_size=2):
    """
    对filter_clusters_by_arc_length的结果进行进一步的IOU合并。
    对于仍有多个候选的簇，基于pairwise IOU合并。
    
    :param filter_results: filter_clusters_by_arc_length的输出结果 (list of (ArcFitResult or list of ArcFitResult))
    :param iou_threshold: IOU阈值，默认0.9
    :param min_group_size: 最小组大小，默认2
    :return: results (list)，每个元素对应一个簇的最佳圆 (ArcFitResult) 或整个cluster
    """