#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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
from sklearn.metrics.pairwise import pairwise_distances_argmin
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 500)
pd.option_context('display.float_format', lambda x: '%.3f' % x)

######################################
# 此处进行了为数据打标签，没有利用循环
def make_label():
    df = pd.read_csv(r'groupedby\39_7.csv')  # 由于csv文件中本来就有列名，故这里不需要header=None
    df_selecet = df.loc[:, ['Cur','Vol']]
    scaler = StandardScaler().fit(df_selecet)
    df_pro = scaler.transform(df_selecet)

    cls = KMeans(n_clusters=4, init='k-means++')    # 分为4类
    kmeans = cls.fit(df_pro)
    # joblib.dump(kmeans, 'kmeans_39_7.model')
    # KMEANS = joblib.load('kmeans_39_7.model')
    # y = KMEANS.predict(df_pro)
    y = kmeans.predict(df_pro)
    # df['Label'] = y[:]
    center = kmeans.cluster_centers_
    actual = np.array([(0.0, 0.0), (0.0, 1.0), (3.6, 1.0), (11.0, 1.0)])
    order = pairwise_distances_argmin(actual, center, axis=1, metric='euclidean')
    # print order
    # print kmeans.cluster_centers_
    # print df.Label.value_counts()
    # print df[df.Label == 0].head()
    # print df[df.Label == 1].head()
    # print df[df.Label == 2].head()
    # print df[df.Label == 3].head()

    # 对于无监督的学习，需要把顺序做对
    n_sample = y.size  # 根据order变换次序
    n_types = 4
    change = np.empty((n_types, n_sample), dtype=np.bool)
    for i in range(n_types):
        change[i] = y == order[i]
    for i in range(n_types):
        y[change[i]] = i
    df['Label'] = y[:]
    df.to_csv(r'groupedby\39_7_label.csv', index=False)

##########################################
# 简单分析测试
# 37_3号设备的电流一直处于较低的状态，其实分为两类或三类都也可以
# df = pd.read_csv(r'groupedby\22_1_label.csv',index_col='Time', parse_dates=['Time'])
# df = pd.read_csv(r'groupedby\37_3_label.csv')
# print df.Label.value_counts()
# print df[df.Label==1]
# print df[df.Label==0]
# print df[df.Label==2]
# print df[df.Label==2]
# print df['2016-12-17 08:27:06']
# with pd.option_context('display.float_format', lambda x: '%.3f' % x):
#     print df['2016-12-17'][df.Label == 0].ix[1]
# print df['2016-12-22'][df.Label == 3]
# print df['2016-12-17'][df.Label == 2]

# print df['2016-12-17'][df.Label == 3].resample('1min').mean()

# print df['2016-12-17'][df.Label == 0].ix[1:5]

###########################################################
# 开始正式处理
def first_feature():
    fileList = listdir('groupedby')
    processedFileList = []
    for i in range(len(fileList)):
        if fileList[i].find('label') > 0:   # 不能单纯用if fileList[i].find('label')，因为结果为-1仍然是代表有值，判断为ture
            processedFileList.append(fileList[i])
    for file_i in processedFileList:
        input_path = 'groupedby\\' + file_i
        output_path = 'label\\' + file_i.replace("_label", "")
        labels = [3,2,0,1]
        days = ['2016-12-17','2016-12-18','2016-12-19','2016-12-20','2016-12-21','2016-12-22','2016-12-23','2016-12-24',
                '2016-12-25','2016-12-26','2016-12-27','2016-12-28','2016-12-29','2016-12-30','2016-12-31','2017-01-01',
                '2017-01-02','2017-01-03','2017-01-04','2017-01-05','2017-01-06','2017-01-07','2017-01-08','2017-01-09',
                '2017-01-10']
        df = pd.read_csv(input_path, index_col='Time', parse_dates=['Time'])
        f = open(output_path, 'a')
        for day_i in days:
            for label_i in labels:
                df_selected = df[day_i][df.Label == label_i].resample('1min').mean()
                begin = []
                item = []
                flag = 0   # flag用来控制连续的NaN
                for i in range(len(df_selected)):
                    if df_selected.ix[i].isnull().any() and flag == 0:
                        flag = 1
                        item.append(time)
                        begin.append(item)
                        item = []
                    elif df_selected.ix[i].isnull().any() and flag == 1:
                        continue
                    elif i == (len(df_selected) - 1):  # 如果是最后一行
                        time = str(df_selected.ix[i].name)
                        item.append(time)
                        begin.append(item)
                        item = []
                    else:  # 如果不是空
                        flag = 0
                        time = str(df_selected.ix[i].name)
                        if len(item) == 0:
                            item.append(time)
                f.write(day_i + ',' + str(label_i) + ',')
                f.write(",".join(map(str, begin)))
                f.write('\n')
        f.close()


####################################################
# 测试用代码
# df = pd.read_csv(r'groupedby\22_1_label.csv',index_col='Time', parse_dates=['Time'])
# df_selected = df['2016-12-17'][df.Label == 2].resample('1min').mean()
# print df_selected.ix[0].name
# print df_selected
# begin = []
# item = []
# for i in range(len(df_selected)):
#     label = df_selected.ix[i]['Label']
#
#     if df_selected.ix[i].isnull().any():
#         item.append(time)
#         begin.append(item)
#         item = []
#     elif i == (len(df_selected)-1): # 如果是最后一行
#         time = df_selected.ix[i].name
#         item.append(time)
#         begin.append(item)
#         item = []
#     else:   # 如果不是空
#         time = df_selected.ix[i].name
#         if len(item) == 0:
#             item.append(time)
#
# print begin
# print df['2016-12-17'][df.Label == 3].resample('1min').mean()

def second_feature():
    days = ['2016-12-17', '2016-12-18', '2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22', '2016-12-23',
            '2016-12-24',
            '2016-12-25', '2016-12-26', '2016-12-27', '2016-12-28', '2016-12-29', '2016-12-30', '2016-12-31',
            '2017-01-01',
            '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-01-08',
            '2017-01-09',
            '2017-01-10']
    # fileList = listdir('groupedby')
    # processedFileList = []
    # for i in range(len(fileList)):
    #     if fileList[i].find('label') > 0:   # 不能单纯用if fileList[i].find('label')，因为结果为-1仍然是代表有值，判断为ture
    #         processedFileList.append(fileList[i])
    # for i in processedFileList:
    #     output_path = "label_2" + i
    df = pd.read_csv(r'groupedby\22_1_label.csv', index_col='Time', parse_dates=['Time'])
    # for day_i in days:
    #     df_select = df[day_i]['Label'].resample('8H', closed='left', label='left').mean()
    #     df_select.to_csv(r'label_2\22_1_8.csv',mode='a')
    # for day_i in days:
    #     df_select = df[day_i]['Label'].resample('6H', closed='left', label='left').mean()
    #     df_select.to_csv(r'label_2\22_1_6H.csv',mode='a')
    for day_i in days:
        df_select = df[day_i]['Label'].resample('1H', closed='left', label='left').mean()
        # 由于1月10号的数据不全，所以人工进行了填充
        df_select.to_csv(r'label_2\22_1_1H.csv',mode='a')

def analysis_firstlabel():
    f = open(r'label\22_1_result.csv', 'a')
    with open(r'label\22_1.csv', 'r') as fs:
        for line in fs:
            line = line.replace("[", "").replace("]", "").replace("'", "").replace(", ",",")
            col = line.strip().split(',')#;print col;exit(0);['2016-12-17', '3', '2016-12-17 08:27:00', '2016-12-17 08:42:00', '2016-12-17 08:44:00', '2016-12-17 08:47:00', '2016-12-17 08:49:00', '2016-12-17 08:52:00', '2016-12-17 08:54:00', '2016-12-17 08:55:00']
            delta = 0
            if len(col) % 2 == 0 :
                L = range(len(col))
                for i in L[2:len(col):2]:
                    t1 = datetime.strptime(col[i], '%Y-%m-%d %H:%M:%S')
                    t2 = datetime.strptime(col[i+1], '%Y-%m-%d %H:%M:%S')
                    diff = (t2-t1).seconds / 60
                    delta += diff
                f.write(col[0] + '\t' + col[1] + '\t' + str(delta))
                f.write('\n')
            elif col[2] == '':
                f.write(col[0] + '\t' + col[1] + '\t' + str(0))
                f.write('\n')
            else:   # 为基数且单独的在最后一个位置
                L = range((len(col)-1))
                for i in L[2: len(col)-1 :2]:
                    t1 = datetime.strptime(col[i], '%Y-%m-%d %H:%M:%S')
                    t2 = datetime.strptime(col[i + 1], '%Y-%m-%d %H:%M:%S')
                    diff = (t2 - t1).seconds / 60
                    delta += diff
                delta += 1
                f.write(col[0] + '\t' + col[1] + '\t' + str(delta))
                f.write('\n')
    f.close()

def make_feature():
    no3 = []
    no2 = []
    no0 = []
    no1 = []
    t8_0 = []
    t8_8 = []
    t8_16 = []
    t6_0 = []
    t6_6 = []
    t6_12 = []
    t6_18 = []
    t1_0 = []
    t1_1 = []
    t1_2 = []
    t1_3 = []
    t1_4 = []
    t1_5 = []
    t1_6 = []
    t1_7 = []
    t1_8 = []
    t1_9 = []
    t1_10 = []
    t1_11 = []
    t1_12 = []
    t1_13 = []
    t1_14 = []
    t1_15 = []
    t1_16 = []
    t1_17 = []
    t1_18 = []
    t1_19 = []
    t1_20 = []
    t1_21 = []
    t1_22 = []
    t1_23 = []

    dic = {
        'no3': no3,
        'no2': no2,
        'no0': no0,
        'no1': no1,
        't8_0': t8_0,
        't8_8': t8_8,
        't8_16': t8_16,
        't6_0': t6_0,
        't6_6': t6_6,
        't6_12': t6_12,
        't6_18': t6_18,
        't1_0': t1_0,
        't1_1': t1_1,
        't1_2': t1_2,
        't1_3': t1_3,
        't1_4': t1_4,
        't1_5': t1_5,
        't1_6': t1_6,
        't1_7': t1_7,
        't1_8': t1_8,
        't1_9': t1_9,
        't1_10': t1_10,
        't1_11': t1_11,
        't1_12': t1_12,
        't1_13': t1_13,
        't1_14': t1_14,
        't1_15': t1_15,
        't1_16': t1_16,
        't1_17': t1_17,
        't1_18': t1_18,
        't1_19': t1_19,
        't1_20': t1_20,
        't1_21': t1_21,
        't1_22': t1_22,
        't1_23': t1_23,
    }

    df_no = pd.read_csv(r'label\22_1_result.csv', header= None , sep= '\t')
    no3.extend(df_no.ix[::4, [2]].values.flatten().tolist())
    no2.extend(df_no.ix[1::4, [2]].values.flatten().tolist())
    no0.extend(df_no.ix[2::4, [2]].values.flatten().tolist())
    no1.extend(df_no.ix[3::4, [2]].values.flatten().tolist())

    df_t8 = pd.read_csv(r'label_2\22_1_8H.csv', header= None)
    t8_0.extend(df_t8.ix[::3, [1]].values.flatten().tolist())
    t8_8.extend(df_t8.ix[1::3, [1]].values.flatten().tolist())
    t8_16.extend(df_t8.ix[2::3, [1]].values.flatten().tolist())

    df_t6 = pd.read_csv(r'label_2\22_1_6H.csv', header= None)
    t6_0.extend(df_t6.ix[::4, [1]].values.flatten().tolist())
    t6_6.extend(df_t6.ix[1::4, [1]].values.flatten().tolist())
    t6_12.extend(df_t6.ix[2::4, [1]].values.flatten().tolist())
    t6_18.extend(df_t6.ix[3::4, [1]].values.flatten().tolist())

    df_t1 = pd.read_csv(r'label_2\22_1_1H.csv', header= None)
    t1_0.extend(df_t1.ix[::24, [1]].values.flatten().tolist())
    t1_1.extend(df_t1.ix[1::24, [1]].values.flatten().tolist())
    t1_2.extend(df_t1.ix[2::24, [1]].values.flatten().tolist())
    t1_3.extend(df_t1.ix[3::24, [1]].values.flatten().tolist())
    t1_4.extend(df_t1.ix[4::24, [1]].values.flatten().tolist())
    t1_5.extend(df_t1.ix[5::24, [1]].values.flatten().tolist())
    t1_6.extend(df_t1.ix[6::24, [1]].values.flatten().tolist())
    t1_7.extend(df_t1.ix[7::24, [1]].values.flatten().tolist())
    t1_8.extend(df_t1.ix[8::24, [1]].values.flatten().tolist())
    t1_9.extend(df_t1.ix[9::24, [1]].values.flatten().tolist())
    t1_10.extend(df_t1.ix[10::24, [1]].values.flatten().tolist())
    t1_11.extend(df_t1.ix[11::24, [1]].values.flatten().tolist())
    t1_12.extend(df_t1.ix[12::24, [1]].values.flatten().tolist())
    t1_13.extend(df_t1.ix[13::24, [1]].values.flatten().tolist())
    t1_14.extend(df_t1.ix[14::24, [1]].values.flatten().tolist())
    t1_15.extend(df_t1.ix[15::24, [1]].values.flatten().tolist())
    t1_16.extend(df_t1.ix[16::24, [1]].values.flatten().tolist())
    t1_17.extend(df_t1.ix[17::24, [1]].values.flatten().tolist())
    t1_18.extend(df_t1.ix[18::24, [1]].values.flatten().tolist())
    t1_19.extend(df_t1.ix[19::24, [1]].values.flatten().tolist())
    t1_20.extend(df_t1.ix[20::24, [1]].values.flatten().tolist())
    t1_21.extend(df_t1.ix[21::24, [1]].values.flatten().tolist())
    t1_22.extend(df_t1.ix[22::24, [1]].values.flatten().tolist())
    t1_23.extend(df_t1.ix[23::24, [1]].values.flatten().tolist())

    df = pd.DataFrame(dic, index=['2016-12-17', '2016-12-18', '2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22', '2016-12-23',
            '2016-12-24',
            '2016-12-25', '2016-12-26', '2016-12-27', '2016-12-28', '2016-12-29', '2016-12-30', '2016-12-31',
            '2017-01-01',
            '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-01-08',
            '2017-01-09',
            '2017-01-10'])
    df.to_csv(r'label_2\label_22_all.csv')

def cluster():
    df = pd.read_csv(r'label_2\label_22_1_all.csv')
    df = df.ix[:,1:].fillna(0)
    df_selecet = df.loc[:, :]
    # print df_selecet;exit(0)
    # scaler = StandardScaler().fit(df_selecet)
    # df_pro = scaler.transform(df_selecet)
    #
    # cls = KMeans(n_clusters=4, init='k-means++')  # 分为4类
    # kmeans = cls.fit(df_pro)
    # joblib.dump(kmeans, 'kmeans_39_7.model')
    # KMEANS = joblib.load('kmeans_39_7.model')
    # y = KMEANS.predict(df_pro)
    # df['Label'] = y[:]
    # df.to_csv(r'groupedby\39_7_label.csv', index=False)

    # df = pd.read_csv(r'cluster_7\processCombined.csv')
    # df = df[np.arange(2)]
    scaler = StandardScaler().fit(df_selecet)
    df_pro = scaler.transform(df_selecet)

    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(df_pro)
        meandistortions.append(
            sum(np.min(cdist(df_pro, kmeans.cluster_centers_, 'euclidean'), axis=1)) / df_pro.shape[0])
        # cdist是求每一个点到每一个聚类中心的欧几里得距离，之后算出每一行最小的距离，然后把所有距离相加，再处理行数，用这里结果来
        # 表示损失函数
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('meandistortions')
    plt.title('Elbow Method')
    plt.show()



if __name__ == "__main__":
    pass
    # make_label()
    # analysis_firstlabel()
    first_feature()
    # second_feature()
    # make_feature()
    # cluster()   # 通过画肘部图可以看出，22号设备不同天数之间毫无规律行
                # 下一步，给出一个完整流程，同时分析25号设备及其它设备的