#!/user/bin/env python
# -*- coding: utf-8 -*-

# 先做一天的（非格式化）
# 选取12月21号   1号设备（NodeAddr = 22） SampleType: 0为电流
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 700)

import matplotlib.pyplot as plt
import matplotlib as mpl
# matplotlib.style.use('ggplot')

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

names = ['DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Data', 'SampleType', 'Time']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# 测试代码
# df = pd.read_csv(r'testdata\1221.csv', names=names, nrows=500, parse_dates=['Time'], date_parser=dateparse)
# print df
# print df.dtypes
# print df.index
# print df.columns
# filt = df[(df.NodeAddr == 22) & (df.SampleType == 0)]
# filt.to_csv(r'currentdata\1221_1.csv', index=False)
# print filt
# df2 = pd.read_csv(r'currentdata\1221_1.csv')  # 再read的时候时间序列又变回到原始类型，而不是时间类型
# print df2
# print df2.dtypes
# print df2.index
# print df2.columns

# 分割文件，包括了再分割为通道文件，但是这里电流没有取均值
# df = pd.read_csv(r'testdata\1221.csv', names=names, parse_dates=['Time'], date_parser=dateparse) # 500万条
# filt = df[(df.NodeAddr == 22) & (df.SampleType == 0)]
# filt.to_csv(r'currentdata\1221_1.csv', index=False)
# df2 = pd.read_csv(r'currentdata\1221_1.csv')
# print df2
# filt_0 = df2[df2.DevChannel == 0]
# filt_0.to_csv(r'currentdata\1221_1_0.csv', index=False) # 12月21号，1号设备，第0通道  的电流数据
# filt_1 = df2[df2.DevChannel == 1]
# filt_1.to_csv(r'currentdata\1221_1_1.csv', index=False)
# filt_2 = df2[df2.DevChannel == 2]
# filt_2.to_csv(r'currentdata\1221_1_2.csv', index=False)

# 电流取均值，生成文件
# df = pd.read_csv(r'currentdata\1221_1.csv', parse_dates=['Time'], date_parser=dateparse)
# df_1 = df.iloc[1::3,:].copy() # 不加copy会报错
# x = df['Data'].values
# # print x
# # print x[1::3]
# N = len(x)
# a = np.arange(0,N)
# result = []
# for i in a[1::3]:
#     b = (x[i-1]+x[i]+x[i+1])/3
#     result.append(b)
# df_1['Data'] = result[:]
# df_1.to_csv(r'currentdata\1221_1_mean.csv', index=False)

# df_plt_0 = pd.read_csv(r'currentdata\1221_1_mean.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
# df_plt_0['Data'].plot()
# plt.show()

# 简单画图
# df_plt_1 = pd.read_csv(r'currentdata\1221_1_1.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
# df_plt_1['Data'].plot(color = 'r',style = '--', lw=1, label=u'原始数据')
# # print df_plt
# plt.grid(b=True)
# plt.show()

# df_plt_0 = pd.read_csv(r'currentdata\1221_1_0.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
# df_plt_0['Data'].plot()
# plt.show()
#
# df_plt_2 = pd.read_csv(r'currentdata\1221_1_2.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
# df_plt_2['Data'].plot()
# plt.show()

# plt.figure(1)
# df_plt_1 = pd.read_csv(r'currentdata\1221_1_1.csv', parse_dates=['Time'], date_parser=dateparse)
# plt.plot(df_plt_1['Time'],df_plt_1['Data'])
#
# plt.figure(2)
# df_plt_11 = pd.read_csv(r'currentdata\1221_1_1.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
# df_plt_11['Data'].plot()
#
# plt.show()

# 异常值处理  异常值处理(即去噪)之后再分析动作，这样的逻辑是正确的！
# def difference(left, right, on):
#     """
#     difference of two dataframes
#     :param left: left dataframe
#     :param right: right dataframe
#     :param on: join key
#     :return: difference dataframe
#     """
#     df = pd.merge(left, right, how='left', on=on)
#     left_columns = left.columns
#     col_y = df.columns[left_columns.size]
#     df = df[df[col_y].isnull()]
#     df = df.ix[:, 0:left_columns.size]
#     df.columns = left_columns
#     return df

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
#
# df_plt_1 = pd.read_csv(r'currentdata\1221_1_1.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
# x = df_plt_1['Data'].values
# # print x
# width = 130 # 采样点应大于噪声的最大宽度
# delta = 10 # 步长应小于噪声的最小宽度
# eps = 3
# N = len(x)
# # p = []
# abnormal = []
# for i in np.arange(0, N-width, delta):
#     s = x[i:i+width]
#     # p.append(np.ptp(s))
#     if np.ptp(s) >eps:
#         abnormal.append(range(i, i+width))
# abnormal = np.array(abnormal).flatten()
# abnormal = np.unique(abnormal)              # abnormal 是不正常值的index
# # plt.plot(p,lw = 1) # 此步确定阈值为3
# # plt.grid(b=True)
# # plt.show()
# df_plt_1_abnormal = df_plt_1.iloc[abnormal]
# # df_plt_1_normal = difference(df_plt_1, df_plt_1_abnormal, 'DataId')
#
# plt.figure(figsize=(18, 7), facecolor='w')
# plt.subplot(131)
# df_plt_1['Data'].plot(color = 'r', lw=1, label=u'原始数据')
# plt.title(u'实际电流数据', fontsize=18)
# plt.legend(loc='upper right')
# plt.grid(b=True)
#
# plt.subplot(132)
# df_plt_1['Data'].plot(color = 'r', lw=1, label=u'原始数据')
# df_plt_1_abnormal['Data'].plot(color = 'g', marker = 'o', markeredgecolor='g', ms=3, label=u'异常值')
# # df_plt_1_abnormal['Data'].plot(color = 'g', style = 'o',label=u'异常值')
# plt.legend(loc='upper right')
# plt.title(u'异常检测', fontsize=18)
# plt.grid(b=True)
#
# plt.subplot(133)
# select = np.ones(N, dtype=np.bool)
# select[abnormal] = False
# t = np.arange(N)
# dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
# br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
# br.fit(t[select].reshape(-1, 1), x[select])
# y = br.predict(np.arange(N).reshape(-1, 1))
# y[select] = x[select]
# # plt.plot(x, 'g--', lw=1, label=u'原始值')  # 原始值
# df_plt_1['Data'].plot(color = 'g', style = '--', lw=1, label=u'原始值')
# # plt.plot(y, 'r-', lw=1, label=u'校正值')  # 校正值
# df_plt_1['Data'] = y[:]
# df_plt_1['Data'].plot(color = 'r', lw=1, label=u'校正值')
# plt.legend(loc='upper right')
# plt.title(u'异常值校正', fontsize=18)
# plt.grid(b=True)
#
# plt.tight_layout(1.5, rect=(0, 0, 1, 0.95))
# plt.suptitle(u'电流数据的异常值检测与校正', fontsize=22)
# plt.show()

# 采用均值文件后的去噪画图，而画图是为了调参，确定width, delta, eps
df_plt_mean = pd.read_csv(r'currentdata\1221_1_mean.csv', parse_dates=['Time'], date_parser=dateparse, index_col='Time')
x = df_plt_mean['Data'].values
# print x
width = 50 # 采样点应大于噪声的最大宽度，需要画出图形实际测试一下
delta = 10 # 步长应小于噪声的最小宽度
eps = 1.5 # 阈值的选取需要实际测试
N = len(x)
p = []
abnormal = []
for i in np.arange(0, N-width, delta):
    s = x[i:i+width]
    p.append(np.ptp(s))
    if np.ptp(s) >eps:
        abnormal.append(range(i, i+width))
abnormal = np.array(abnormal).flatten()
abnormal = np.unique(abnormal)              # abnormal 是不正常值的index
plt.plot(p,lw = 1) # 此步确定阈值为3
p.sort()
print p
plt.grid(b=True)
plt.show()
df_plt_mean_abnormal = df_plt_mean.iloc[abnormal]
print df_plt_mean_abnormal

plt.figure(figsize=(18, 7), facecolor='w')
plt.subplot(131)
df_plt_mean['Data'].plot(color = 'r', lw=1, label=u'原始数据')
plt.title(u'实际电流数据', fontsize=18)
plt.legend(loc='upper right')
plt.grid(b=True)

plt.subplot(132)
df_plt_mean['Data'].plot(color = 'r', lw=1, label=u'原始数据')
df_plt_mean_abnormal['Data'].plot(color = 'g', marker = 'o', markeredgecolor='g', ms=3, label=u'异常值')
plt.legend(loc='upper right')
plt.title(u'异常检测', fontsize=18)
plt.grid(b=True)

plt.subplot(133)
select = np.ones(N, dtype=np.bool)
select[abnormal] = False
t = np.arange(N)
dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
br.fit(t[select].reshape(-1, 1), x[select])
y = br.predict(np.arange(N).reshape(-1, 1))
y[select] = x[select]
df_plt_mean['Data'].plot(color = 'g', style = '--', lw=1, label=u'原始值')
df_plt_mean['Data'] = y[:]
df_plt_mean['Data'].plot(color = 'r', lw=1, label=u'校正值')
plt.legend(loc='upper right')
plt.title(u'异常值校正', fontsize=18)
plt.grid(b=True)
plt.tight_layout(1.5, rect=(0, 0, 1, 0.95))
plt.suptitle(u'电流数据的异常值检测与校正', fontsize=22)
plt.show()
exit(0)

# 去噪
# df_plt_mean = pd.read_csv(r'currentdata\1221_1_mean.csv', parse_dates=['Time'], date_parser=dateparse)
# x = df_plt_mean['Data'].values
# width = 50 # 采样点应大于噪声的最大宽度，需要画出图形实际测试一下
# delta = 10 # 步长应小于噪声的最小宽度
# eps = 1.5 # 阈值的选取需要实际测试
# N = len(x)
# abnormal = []
# for i in np.arange(0, N-width, delta):
#     s = x[i:i+width]
#     if np.ptp(s) >eps:
#         abnormal.append(range(i, i+width))
# abnormal = np.array(abnormal).flatten()
# abnormal = np.unique(abnormal)              # abnormal 是不正常值的index
# # df_plt_mean_abnormal = df_plt_mean.iloc[abnormal]
#
# select = np.ones(N, dtype=np.bool)
# select[abnormal] = False
# t = np.arange(N)
# dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
# br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
# br.fit(t[select].reshape(-1, 1), x[select])
# y = br.predict(np.arange(N).reshape(-1, 1))
# y[select] = x[select]
# df_plt_mean['Data'] = y[:]
# df_plt_mean.to_csv(r'currentdata\1221_1_mean_purebackup.csv', index=False)

#之后遍历每天的数据去做（格式化）
#在all.py文件中