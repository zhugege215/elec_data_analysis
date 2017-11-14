#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cPickle as pickle
from os import listdir
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.externals import joblib
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 500)
pd.option_context('display.float_format', lambda x: '%.3f' % x)

names = ['Time', 'DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Cur', 'Vol']

'''
df = pd.read_csv(r'a_split_bydevice_mean\transformed_removed_28_4.csv', parse_dates=['Time'], index_col='Time')
df = df['2016-12-20 10:00:00':'2016-12-20 11:00:00']
# print df.resample('1S', closed='left', label='left').mean()
df_0 = df[df.DevChannel == 0]
# df_1 = df[df.DevChannel == 1]
# df_2 = df[df.DevChannel == 2]
# df_0 = df_0.resample('1S', closed='left', label='left').mean().ffill()
# df_1 = df_1.resample('1S', closed='left', label='left').mean()
# df_2 = df_2.resample('1S', closed='left', label='left').mean()

# plt.subplot(3,1,1)
# plt.title('0 channel of current')
# df_0['Cur'].plot()
# plt.subplot(3,1,2)
# plt.title('1 channel of current')
# df_1['Cur'].plot()
# plt.subplot(3,1,3)
# plt.title('2 channel of current')
# df_2['Cur'].plot()
# plt.tight_layout()
# plt.show()

# 通过多个时间段比较（比如7到8点，8到9点，9到10点），无论是resample的还是不resample的，都发现三个通道的
# 变化趋势相近，故分析其中的一个即可。且0通道的噪声最小，故选择0。

# df_0 = df_0.resample('1S', closed='left', label='left').mean().ffill()   # 用这种方式
df_0['Cur'].plot()      #  10:19:00-10:24:00 有变化，波动
plt.show()              #  13:02-13:08
                        #  13:37-13:45    数据不全！
                        #  14:16-14:51
'''

def find_action():
    # 同样的，用pandas运行太慢
    # 错了，不是用pandas太慢，而是程序有问题，不应该遍历i的同时还用到 i+1 行
    filelist = listdir('a_split_bydevice_mean')
    processed_filelist = []
    for i in range(len(filelist)):
        if filelist[i].find('transformed') == 0:
            processed_filelist.append(filelist[i])
    for filename in processed_filelist:
        df = pd.read_csv('a_split_bydevice_mean\\'+filename, parse_dates=['Time'], index_col='Time')
        df = df[df.DevChannel == 0]
        df_copy = df.copy()
        df_copy.Cur = df_copy.Cur.shift(-1)
        df_1 = df[:-1].copy()
        df_2 = df_copy[:-1].copy()
        x = df_1['Cur'].values
        y = df_2['Cur'].values
        N = len(x)
        abnormal = []
        select = np.zeros(N, dtype=np.bool)
        for i in range(N):
            if abs(y[i]-x[i]) > 2:
                abnormal.append(i)
        select[abnormal] = True
        # print df_1[select]  # 至此找到所有波动的点
        df_1[select].to_csv(r'state_analysis.csv', header=False, mode='a')

if __name__ == "__main__":
    # # find_action()

    ##################################找到电流变化的大致时间区间
    # df = pd.read_csv('state_analysis.csv', names=names, index_col='Time', parse_dates=['Time'])
    # df = df[df.NodeAddr == 28]
    # df = df['2016-12-29']
    # print df

    # 一个小脚本，最终将其存储到state_analysis_index.csv文件中
    # df = pd.read_csv('state_analysis.csv', names=names, index_col='Time', parse_dates=['Time'])
    # df = df[df.NodeAddr == 39]  # 没有利用循环
    # days = ['2016-12-17', '2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22', '2016-12-23',
    #         '2016-12-24',
    #         '2016-12-26', '2016-12-27', '2016-12-28', '2016-12-29', '2016-12-30', '2016-12-31',
    #         '2017-01-01',
    #         '2017-01-02', '2017-01-03']
    # f = open('state_analysis_index.csv', 'a')
    # for day_i in days:
    #     df_select = df[day_i].resample('1min').mean()
    #     begin = []
    #     item = []
    #     flag = 0  # flag用来控制连续的NaN
    #     for i in range(len(df_select)):
    #         if df_select.ix[i].isnull().any() and flag == 0:
    #             flag = 1
    #             item.append(time)
    #             begin.append(item)
    #             item = []
    #         elif df_select.ix[i].isnull().any() and flag == 1:
    #             continue
    #         elif i == (len(df_select) - 1):  # 如果是最后一行
    #             time = str(df_select.ix[i].name)
    #             item.append(time)
    #             begin.append(item)
    #             item = []
    #         else:  # 如果不是空
    #             flag = 0
    #             time = str(df_select.ix[i].name)
    #             if len(item) == 0:
    #                 item.append(time)
    #     f.write(day_i + ',')
    #     f.write(",".join(map(str, begin)))
    #     f.write('\n')
    # f.close()



    ##################################对应时间区间找到具体的数值
    # df = pd.read_csv(r'a_split_bydevice_mean\transformed_removed_28_4.csv', index_col='Time', parse_dates=['Time'])
    # df = df[df.DevChannel == 0]
    # print df['2016-12-19 08:54:00':'2016-12-19 08:55:00']

    ##################################对应时间区间画出图分析
    df = pd.read_csv(r'a_split_bydevice_mean\transformed_removed_28_4.csv', index_col='Time', parse_dates=['Time'])
    df = df[df.DevChannel == 0]
    # df['2016-12-29']['Cur'].plot()  # 观察发现，29号这一天电流值都处于一个较低的水平
    df['2016-12-17 08:30:00':'2016-12-17 08:35:00']['Cur'].plot()
    plt.show()
