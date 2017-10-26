# #!/user/bin/env python
# # -*- coding: utf-8 -*-
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
#
pd.set_option('display.width', 200)
names = ['DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Data', 'SampleType', 'Time']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#
# # df = pd.read_csv(r'testdata\1221.csv', names=names, parse_dates=['Time'], date_parser=dateparse,nrows=50)  # 500万条
# # print df
#
# df = pd.read_csv(r'testdata\1221.csv', names=names, parse_dates=['Time'], date_parser=dateparse, index_col='Time')  # 500万条
# filt_1 = df[(df.NodeAddr == 22) & (df.SampleType == 1)] # SampleType取1表示电压，NodeAddr取22表示空1设备
# # print filt_1[:30]
# filt_1_mean = filt_1.iloc[1::3, :].copy()
# x = filt_1['Data'].values
# N = len(x)
# a = np.arange(0,N)
# result = []
# for j in a[1::3]:
#     b = (x[j-1]+x[j]+x[j+1])/3
#     result.append(b)
# filt_1_mean['Data'] = result[:]
# filt_1_mean['Data'].plot()
# plt.grid(b=True)
# plt.show()

# 经过观察，电压数据无需去噪
if __name__ == '__main__':
    fileList = listdir('testdata')
    m = len(fileList)
    for i in range(m):
        fileNameStr = fileList[i]
        day = fileNameStr[0:4]
        fileName = r'testdata\%s.csv' % day

        # 读每天数据的1号设备的电压数据
        df = pd.read_csv(fileName, names=names, parse_dates=['Time'], date_parser=dateparse)
        filt_1 = df[(df.NodeAddr == 22) & (df.SampleType == 1)]  # filt_1 筛选出的这一天的1号设备的电压值


        # 对电压数据取均值
        filt_1_mean = filt_1.iloc[1::3, :].copy()
        x = filt_1['Data'].values
        N = len(x)
        a = np.arange(0, N)
        result = []
        for j in a[1::3]:
            b = (x[j - 1] + x[j] + x[j + 1]) / 3
            result.append(b)
        filt_1_mean['Data'] = result[:]

        to_filename = r'vol_process_1\%s_1_mean.csv' % day  # 将处理后的数据统统放到cur_process_1中去
        filt_1_mean.to_csv(to_filename, index=False)