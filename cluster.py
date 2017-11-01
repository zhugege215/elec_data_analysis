#!/user/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 200)
names = ['DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Data', 'SampleType', 'Time']

# cur = [1,2,3]
# vol = [4,5,6]
# dic = {'cur': cur,
#        'vol': vol}
# df1 = pd.DataFrame(dic)
# print df1
# df1.to_csv('combine_test.csv',columns=['cur','vol'],index=False,header=True,mode='w')
# df2 = pd.DataFrame(dic)
# print df2
# df2.to_csv('combine_test.csv',columns=['vol','cur'],index=False,header=False,mode='a')

# list1 = [1,1]
# df['cur'].(list1)
# # df['vol'].add(list1)
# print df
# df = pd.read_csv('a.csv',names = names,nrows=60)
# print df


# cur = []
# vol = []
# dic = {'cur': cur,
#        'vol': vol}
# # df1 = pd.DataFrame(dic)
# chunker = pd.read_csv('a.csv',names=names,chunksize=9)
# i = 0
# for piece in chunker:
#     i += 1
#     if piece.NodeAddr.iloc[0] == 24 : # 这里注意一定要用iloc。。。
#         a = piece.Data.tolist()
#         print a
#         print a[0]
#         print type(a)
#         print type(a[0])
#     if piece.NodeAddr.iloc[0] == 28 : # 这里注意一定要用iloc。。。
#         a = piece.Data.tolist()
#         print a
#         print a[0]
#         print type(a)
#         print type(a[0])
#         cur.extend(a[:3])
#         vol.extend(a[3:6])
#         df1 = pd.DataFrame(dic)
#         print df1
#     if i ==2:
#         exit(0)

# 这个脚本的思路非常好，但是实践中发现，对于太大的数据文件就不要用pandas了（比如这里6个g），太吃内存了，还是老老实实读行吧。。。
# def show_method():
#     chunker = pd.read_csv(r'a.csv',names=names,chunksize=9)
#     for piece in chunker:
#         if piece.NodeAddr.iloc[0] == 22:
#             cur = []
#             vol = []
#             dic = {'cur': cur,
#                    'vol': vol}
#             data = piece.Data.tolist()
#             cur.extend(data[:3])
#             vol.extend(data[3:6])
#             df = pd.DataFrame(dic)
#             df.to_csv(r'combine_test.csv',columns=['cur','vol'],index=False,header=False,mode='a') # columns确保写入的顺序正确
#         else:
#             pass
#     return None
#
# if __name__ == '__main__':
#     show_method()


############################################################
# # 分割文件
# f_cur = open(r'a_22_cur.csv', 'a')
# f_vol = open(r'a_22_vol.csv', 'a')
#
# operation = {
#     '00220': lambda data: f_cur.write(data+'\n'),
#     '00221': lambda data: f_vol.write(data+'\n')
# }
#
# def write_file(sample, data):
#     operation.get(sample, lambda x:"nothing" )(data) # 默认为 nothing，但这里不能写为default = 0,否则会报错
#
# f = open(r'a.csv','r')
# i = 0
# for line in f:
#     col = line.strip().split(',')
#     try:
#     # a = str(col[1]) + str(col[5])
#     # print a
#         write_file(str(col[1]) + str(col[5]), col[4])
#     except Exception, e:
#         print Exception, ":", e
#         print 'error rows = ', i    # 同样是最后一行出错
#     i += 1
#
# f.close()
# f_cur.close()
# f_vol.close()

#################################################################
# # 合并电流和电压
# df = pd.read_csv(r'a_22_cur.csv',header=None,nrows=6)
# print df

# cur_22 = pd.read_csv(r'a_22_cur.csv', header=None)
# vol_22 = pd.read_csv(r'a_22_vol.csv', header=None)
#
# feature_notmean = pd.concat([cur_22, vol_22], axis=1)
# # print feature_notmean[:10]
# feature_notmean.to_csv('combine_22_notmean.csv',index=False,header=False)

##################################################################
# # 聚类初步
# # notmean 的前提是默认三通道是相互独立的
# # df = pd.read_csv(r'combine_22_notmean.csv',header=None,nrows=6)
# # print df

#
# def expand(a, b):
#     d = (b - a) * 0.1
#     return a-d, b+d
#
# df = pd.read_csv(r'combine_22_notmean.csv',header=None)
# df = df.as_matrix()
# # print df[:10,0];exit(0)
#
# cls = KMeans(n_clusters=3, init='k-means++')
# y = cls.fit_predict(df)
#
# matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# cm = matplotlib.colors.ListedColormap(list('rgbm'))
#
# plt.figure(facecolor='w')
# plt.title(u'KMeans++聚类')
# plt.scatter(df[:, 0], df[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
# x1_min, x2_min = np.min(df, axis=0)
# x1_max, x2_max = np.max(df, axis=0)
# x1_min, x1_max = expand(x1_min, x1_max)
# x2_min, x2_max = expand(x2_min, x2_max)
# plt.xlim((x1_min, x1_max))
# plt.ylim((x2_min, x2_max))
# plt.grid(True)
# plt.show()

#######################################################################
# # 选择文件 + 聚类分析
# f_cur = open(r'cluster\1221_cur.csv', 'a')
# f_vol = open(r'cluster\1221_vol.csv', 'a')
#
# operation = {
#     '00220': lambda data: f_cur.write(data+'\n'),
#     '00221': lambda data: f_vol.write(data+'\n')
# }
#
# def write_file(sample, data):
#     operation.get(sample, lambda x:"nothing" )(data) # 默认为 nothing，但这里不能写为default = 0,否则会报错
#
# f = open(r'testdata\1221.csv','r')
# i = 0
# for line in f:
#     col = line.strip().split(',')
#     try:
#     # a = str(col[1]) + str(col[5])
#     # print a
#         write_file(str(col[1]) + str(col[5]), col[4])
#     except Exception, e:
#         print Exception, ":", e
#         print 'error rows = ', i    # 同样是最后一行出错
#     i += 1
#
# f.close()
# f_cur.close()
# f_vol.close()
#
#
# cur_22 = pd.read_csv(r'cluster\1221_cur.csv', header=None)
# vol_22 = pd.read_csv(r'cluster\1221_vol.csv', header=None)
#
# feature_notmean = pd.concat([cur_22, vol_22], axis=1)
# feature_notmean.to_csv(r'cluster\combine_1221.csv',index=False,header=False)


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

df = pd.read_csv(r'cluster\combine_1221.csv', header=None)
# df = df.as_matrix()
df = df[np.arange(2)]
# print df[[0]];exit(0)

N, M = 1000, 1000     # 横纵各采样多少个值
x1_min, x2_min = df.min()
x1_max, x2_max = df.max()
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
x_grid = np.stack((x1.flat, x2.flat), axis=1)   # 测试点
# x_pro = scale(x_grid)
exit(0)
#
# df_pro = scale(df)  # 电流的量级是10左右，电压是200左右，不归一化是有问题的，因为 kmeans 聚类本质是计算点的距离
# 这里用 StandardScaler 也是可以的

scaler = StandardScaler().fit(df)
df_pro = scaler.transform(df)
x_pro = scaler.transform(x_grid)

cls = KMeans(n_clusters=4, init='k-means++')
# y = cls.fit_predict(df_pro)
cls.fit(df_pro)
y = cls.predict(df_pro)
y_unique = np.unique(y)
print y_unique


matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
cm = matplotlib.colors.ListedColormap(list('rgbm'))

cm_light = matplotlib.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF', '#BC8F8F'])
print cls.cluster_centers_
y_grid_hat = cls.predict(x_pro)
print cls.cluster_centers_
y_grid_hat_unique = np.unique(y_grid_hat)
print y_grid_hat_unique

y_grid_hat = y_grid_hat.reshape(x1.shape)

plt.figure(facecolor='w')
plt.title(u'KMeans++聚类')
plt.pcolormesh(x1, x2, y_grid_hat, cmap=cm_light)
plt.scatter(df[[0]], df[[1]], c=y, s=30, cmap=cm, edgecolors='none')
# x1_min, x2_min = np.min(df, axis=0)
# x1_max, x2_max = np.max(df, axis=0)
# x1_min, x1_max = expand(x1_min, x1_max)
# x2_min, x2_max = expand(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.grid(True)
plt.show()

#######################################################################
# # 选择18(Sun) - 24(Sat)号22号设备数据




