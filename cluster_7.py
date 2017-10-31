#!/user/bin/env python
# -*- coding: utf-8 -*-
# # 选择18(Sun) - 24(Sat)号22号设备数据
# 选取 12.18-12.24 这7天，设备编号为 0024 的电流值和电压值

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def splitFile():
    f_cur = open(r'cluster_7\cur_7_22.csv', 'a')
    f_vol = open(r'cluster_7\vol_7_22.csv', 'a')
    f = open(r'a.csv', 'r')
    operation = {
        '2016-12-1800220': lambda data: f_cur.write(data + '\n'),
        '2016-12-1800221': lambda data: f_vol.write(data + '\n'),
        '2016-12-1900220': lambda data: f_cur.write(data + '\n'),
        '2016-12-1900221': lambda data: f_vol.write(data + '\n'),
        '2016-12-2000220': lambda data: f_cur.write(data + '\n'),
        '2016-12-2000221': lambda data: f_vol.write(data + '\n'),
        '2016-12-2100220': lambda data: f_cur.write(data + '\n'),
        '2016-12-2100221': lambda data: f_vol.write(data + '\n'),
        '2016-12-2200220': lambda data: f_cur.write(data + '\n'),
        '2016-12-2200221': lambda data: f_vol.write(data + '\n'),
        '2016-12-2300220': lambda data: f_cur.write(data + '\n'),
        '2016-12-2300221': lambda data: f_vol.write(data + '\n'),
        '2016-12-2400220': lambda data: f_cur.write(data + '\n'),
        '2016-12-2400221': lambda data: f_vol.write(data + '\n'),
    }
    i = 0
    for line in f:
        col = line.strip().split(',')
        try:
            operation.get(col[6].split(' ')[0] + str(col[1]) + str(col[5]), lambda x: None)(col[4])
        except Exception, e:
            print Exception, ':', e
            print 'error rows = ', i
        i += 1
    f.close()
    f_vol.close()
    f_cur.close()

def dataProcess():
    # 对电流、电压数据取均值
    cur_7 = pd.read_csv(r'cluster_7\cur_7_22.csv', header=None)
    data_cur = cur_7[[0]].values
    N = len(data_cur)
    a = np.arange(0,N)
    cur_mean = []
    for i in a[1::3]:
        b = (data_cur[i-1] + data_cur[i] + data_cur[i+1])/3
        cur_mean.extend(b)  # 注意这里要用extend

    vol_7 = pd.read_csv(r'cluster_7\vol_7_22.csv', header=None)
    data_vol = vol_7[[0]].values
    vol_mean = []
    for i in a[1::3]:
        b = (data_vol[i-1] + data_vol[i] + data_vol[i+1])/3
        vol_mean.extend(b)  # 注意这里要用extend

    # 对电流数据进行去噪, 电压数据暂不需要
    cur_mean = np.array(cur_mean).reshape(-1, 1) # 变为1列
    width = 50
    delta = 10
    eps = 1.2
    NN = len(cur_mean)
    abnormal = []
    for k in np.arange(0, NN - width, delta):
        s = cur_mean[k:k + width]
        if np.ptp(s) > eps:
            abnormal.append(range(k, k + width))
    abnormal = np.array(abnormal).flatten()
    abnormal = np.unique(abnormal)
    abnormal = np.array(abnormal, dtype=int)

    select = np.ones(NN, dtype=np.bool)
    select[abnormal] = False
    t = np.arange(NN)
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
    br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
    br.fit(t[select].reshape(-1, 1), cur_mean[select].ravel())  # 由警告改成.ravel()
    cur_mean_pure = br.predict(np.arange(NN).reshape(-1, 1))
    cur_mean_pure[select] = cur_mean[select]
    print type(cur_mean_pure)

    d = {'cur': cur_mean_pure.flatten().tolist(),
         'vol': vol_mean}
    df = pd.DataFrame(d)
    df.to_csv(r'cluster_7\processCombined.csv', index=False)

def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

def cluster(numOfCluster=4):    # 默认分为4类
    df = pd.read_csv(r'cluster_7\processCombined.csv')  # 由于csv文件中本来就有列名，故这里不需要header=None
    df = df[np.arange(2)]
    # print df[0];exit(0)

    N, M = 1000, 1000  # 横纵各采样多少个值
    x1_min, x2_min = df.min()
    x1_max, x2_max = df.max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_grid = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    scaler = StandardScaler().fit(df)
    df_pro = scaler.transform(df)
    x_pro = scaler.transform(x_grid)

    cls = KMeans(n_clusters=numOfCluster, init='k-means++')  # 此处用到参数
    # y = cls.fit_predict(df_pro)
    cls.fit(df_pro)
    print 'the cluster centers are:\n', cls.cluster_centers_
    y = cls.predict(df_pro)
    y_unique = np.unique(y)
    print 'the term of y are:', y_unique

    y_grid_hat = cls.predict(x_pro)
    y_grid_hat_unique = np.unique(y_grid_hat)
    print 'the term of y_grid_hat are:', y_grid_hat_unique

    y_grid_hat = y_grid_hat.reshape(x1.shape)

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))
    cm_light = matplotlib.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF', '#BC8F8F'])

    plt.figure(facecolor='w')
    plt.title(u'KMeans++聚类')
    plt.pcolormesh(x1, x2, y_grid_hat, cmap=cm_light)
    plt.scatter(df[[0]], df[[1]], c=y, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)
    plt.show()

def ppp():
    print 'test'

bianliang = 1314

def elbowMethod():
    df = pd.read_csv(r'cluster_7\processCombined.csv')
    df = df[np.arange(2)]
    scaler = StandardScaler().fit(df)
    df_pro = scaler.transform(df)

    K = range(1,10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(df_pro)
        meandistortions.append(sum(np.min(cdist(df_pro, kmeans.cluster_centers_, 'euclidean'), axis=1))/df_pro.shape[0])
        # cdist是求每一个点到每一个聚类中心的欧几里得距离，之后算出每一行最小的距离，然后把所有距离相加，再处理行数，用这里结果来
        # 表示损失函数
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('meandistortions')
    plt.title('Elbow Method')
    plt.show()

if __name__ == "__main__":
    # splitFile()
    # dataProcess()
    cluster(3)   # default: numOfCluster=4
    # elbowMethod()
    # df = pd.read_csv(r'cluster_7\vol_7_22.csv',header=None)
    # print df.describe()
    # dfq = pd.read_csv(r'cluster_7\cur_7_22.csv',header=None)
    # print dfq.describe()
    # dfq = pd.read_csv(r'a.csv', header=None, nrows=50)
    # print dfq

    # df = pd.read_csv(r'cluster_7\processCombined.csv')
    # plt.scatter(df['cur'],df['vol'])
    # plt.show()
    # print df
    # print df.dtypes
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(df['vol'])
    # plt.title('vol')
    # plt.subplot(212)
    # plt.plot(df['cur'])
    # plt.title('cur')
    # plt.show()


    # 验证dataprocess正不正确，直接画出电流、电压值的图即可


