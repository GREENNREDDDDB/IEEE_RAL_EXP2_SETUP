from sklearn.pipeline import Pipeline
import numpy as np

def curve_3d(t_range: np.ndarray, poly_reg_A, poly_reg_B):

    dt = 0.001  # 变化率
    t = np.arange(t_range.min(), t_range.max(), dt)
    x = t
    x_column = x.reshape(-1, 1)
    y = poly_reg_A.predict(x_column)
    z = poly_reg_B.predict(x_column)

    points = np.array([x,y,z]).transpose()

    area_list = [] # 存储每一微小步长的曲线长度

    for i in range(1,len(t)):
        # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
        dl = np.linalg.norm(points[i] - points[i-1])
        #dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2 )
        # 将计算结果存储起来
        area_list.append(dl)

    length = np.sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度


    return length
