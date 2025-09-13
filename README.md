# 重耋圆瞪眼算法
尝试从多个重叠圆形中找到交点，从而划分并拟合出不同的圆

## 处理流程步骤

1. **生成样本图像**
   - 在白色背景生成四个相交的黑色实心圆形，圆心在矩形的四个角。

2. **图像预处理**
   - 灰度化、高斯滤波、二值化。

3. **获取轮廓**
   - 提取所有轮廓，筛选出重叠圆的外轮廓和内轮廓，排除图像边框轮廓。

4. **B样条拟合轮廓**
   - 对每条轮廓进行B样条拟合（如s=20），并采样200个点。

<a href="https://ibb.co/zytDjFZ"><img src="https://i.ibb.co/XhGMNDt/Figure-1.png" alt="Figure-1" border="0"></a>

5. **寻找圆弧交点（凹点）**
   - 用局部向量夹角法找出轮廓上的凹点（交点），并过滤距离过近的点。

<a href="https://ibb.co/XTf3vy7"><img src="https://i.ibb.co/t5wxWZq/Figure-2.png" alt="Figure-2" border="0"></a>

6. **切割样条，采样得到新圆弧**
   - 根据凹点分割样条曲线，采样得到各个圆弧段。

7. **拟合圆弧为圆**
   - 对每个圆弧段拟合圆，得到圆心、半径和弧长。

<a href="https://ibb.co/3mNnkpjW"><img src="https://i.ibb.co/SwnzJtM0/Figure-3.png" alt="Figure-3" border="0"></a>

8. **圆形分簇与筛选**
   - 先根据与黑色区域的IOU过滤圆形，再用贪心算法分簇。
   - 对每个簇应用弧长过滤，采纳可信的拟合圆。

9. **进一步合并候选簇**
   - 对仍有多个候选的簇，基于簇内圆的IOU进一步合并，最终得到最佳圆形结果。

<a href="https://ibb.co/sdcSbc8n"><img src="https://i.ibb.co/FLcvmcR2/Figure-4.png" alt="Figure-4" border="0"></a>

---
详细接口和参数见 [interface.md](interface.md)。