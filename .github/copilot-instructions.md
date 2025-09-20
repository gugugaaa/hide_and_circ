# Copilot Instructions for hide_and_circ

## 项目架构与主要流程
- 本项目旨在从多个重叠圆形图像中分离并拟合出各个圆，核心流程见 `README.md`。
- 主要处理流程：样本生成 → 图像预处理 → 轮廓提取 → B样条拟合 → 凹点检测 → 样条切割 → 圆拟合 → 分簇与筛选 → 合并候选。
- 关键模块均在 `utils/` 目录下，按功能拆分：如 `fit_circle.py`（圆拟合）、`get_contours.py`（轮廓提取）、`concave_pts.py`（凹点检测）、`merge_by_iou.py`（IOU合并与筛选）。
- `main.py` 为主入口，负责流程调度和模块调用。

## 代码风格与约定
- 各功能模块独立，函数命名以动词+对象为主（如 `fit_circle`, `get_contours`）。
- 图像处理流程严格按步骤分层，避免跨模块耦合。
- 采样点数量、B样条参数等均为可调参数，建议通过主入口传递。
- 结果数据结构以 numpy 数组和 dict 为主，便于后续处理。

## 开发与调试
- 依赖管理：所有依赖写在 `requirements.txt`，如需新增请同步更新。
- 推荐使用 `playground/draw_sample.py` 进行样本生成和算法调试。
- 主要调试入口为 `main.py`，可直接运行。
- 所有开放接口写在 `interface.md`，可以查看。

## 重要文件/目录说明
- `main.py`：主流程入口，串联各处理步骤。
- `utils/`：算法核心模块，按功能拆分。
- `playground/`：样本生成与调试脚本。
- `README.md`：详细流程说明与算法原理。
- `interface.md`：接口与参数文档。

## 其他约定
- 图像输入输出均采用 numpy 数组，圆参数统一为 (x, y, r)。
- 结果筛选以 IOU 和弧长为主，分簇采用贪心算法。
