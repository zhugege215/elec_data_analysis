#!/user/bin/env python
# -*- coding: utf-8 -*-

from os import listdir
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

names = ['DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Data', 'SampleType', 'Time']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

fileList = listdir('testdata')
m = len(fileList)
for i in range(m):
    # print fileList[i]
    # print type(fileList[i]);exit(0)
    fileNameStr = fileList[i]
    # print fileNameStr[0:2];exit(0)
    # print type(fileNameStr[0:2])
    day = fileNameStr[0:4]
    # print day
    # print type(day)
    fileName = r'testdata\%s.csv' % day
    # print fileName;exit(0)

    # 读每天数据的1号设备的电流数据
    df = pd.read_csv(fileName, names=names, parse_dates=['Time'], date_parser=dateparse)  # 500万条
    filt_1 = df[(df.NodeAddr == 22) & (df.SampleType == 0)]  # filt_1 筛选出的这一天的1号设备的电流值

    # print filt_1[:6]
    # b = filt_1[:10]
    # print b.iloc[0:3]

    # 对电流数据取均值
    filt_1_mean = filt_1.iloc[1::3, :].copy()
    x = filt_1['Data'].values
    N = len(x)
    a = np.arange(0,N)
    result = []
    for j in a[1::3]:
        b = (x[j-1]+x[j]+x[j+1])/3
        result.append(b)
    filt_1_mean['Data'] = result[:]

    # 去噪，生成新文件
    xx = filt_1_mean['Data'].values
    width = 50
    delta = 10
    eps = 1.5
    NN = len(xx)
    abnormal = []
    for k in np.arange(0, NN-width, delta):
        s = xx[k:k+width]
        if np.ptp(s) > eps:
            abnormal.append(range(k, k+width))
    abnormal = np.array(abnormal).flatten()
    abnormal = np.unique(abnormal)              # abnormal 是不正常值的index
    # filt_1_mean_abnormal = filt_1_mean.iloc[abnormal]

    # print type(abnormal)
    # print
    # print abnormal;exit(0)
    abnormal = np.array(abnormal, dtype=int)    # 不知所以。。。但是这行代码非常重要，否则会发生错误
    # 但是在分开做的时候没有遇到这类问题

    select = np.ones(NN, dtype=np.bool)
    select[abnormal] = False
    t = np.arange(NN)
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
    br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
    br.fit(t[select].reshape(-1, 1), xx[select])
    y = br.predict(np.arange(NN).reshape(-1, 1))
    y[select] = xx[select]
    filt_1_mean['Data'] = y[:]
    to_filename = r'cur_process_1\%s_1_mean_pure.csv' % day    # 将处理后的数据统统放到cur_process_1中去
    filt_1_mean.to_csv(to_filename, index=False)

# day = '0101'
# s = r'cur_process_1\%s_1_mean_pure.csv' % day
# print s