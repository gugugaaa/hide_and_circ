import numpy as np

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

class ArcFitResult:
    """
    拟合圆弧的结果类
    包含圆心坐标(center_x, center_y), 半径(radius), 以及弧长(arc_length)
    """
    def __init__(self, center_x, center_y, radius, arc_length):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.arc_length = arc_length

def fit_arc_to_circle(seg_points, arc_length):
    """
    根据样条的采样点，拟合圆弧
    返回： ArcFitResult 或 None
    """
    circle_params = fit_circle_least_squares(seg_points)
    if circle_params is not None:
        cx, cy, r = circle_params
        return ArcFitResult(cx, cy, r, arc_length)
    return None