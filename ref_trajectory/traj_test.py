# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from scipy.interpolate import make_interp_spline

# 绘制路点曲率曲线
if __name__ == '__main__':
    #--------------------------------------------- 数据准备 ---------------------------------------------

    log_f_name = "traj_6.csv"
    # log_f_name = "traj_6_reference.csv"
    print("loading data from : " + log_f_name)
    data = pd.read_csv(log_f_name)  # 读取数据
    print("data shape : ", data.shape, "data type : ", type(data))

    #--------------------------------------------- 数据处理 ---------------------------------------------
    x = data['world_x']
    y = data['world_y']
    x_smooth = np.linspace(data['distance'].min(), data['distance'].max(), 100000)   # 插值
    y_smooth = make_interp_spline(data['distance'], data['curvature'])(x_smooth)

    max = 10
    index = 0
    for i in range(len(x)-5):
        d = math.sqrt((x[i+5]-x[i])**2 + (y[i+5]-y[i])**2)
        if d < max:
            max = d
            index = i
    print(max, "    ", index)
