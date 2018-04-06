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
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 500)
pd.option_context('display.float_format', lambda x: '%.3f' % x)

def analysis_firstlabel():
    filelist = ['22_1', '25_2', '28_4', '34_5', '35_6', '37_3', '39_7']
    for filename in filelist:
        outputpath = 'label\\' + filename + '_result.csv'
        inputpath = 'label\\' + filename + '.csv'
        f = open(outputpath, 'a')
        with open(inputpath, 'r') as fs:
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

def second_feature():
    # days = ['2016-12-17', '2016-12-18', '2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22', '2016-12-23',
    #         '2016-12-24',
    #         '2016-12-25', '2016-12-26', '2016-12-27', '2016-12-28', '2016-12-29', '2016-12-30', '2016-12-31',
    #         '2017-01-01',
    #         '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-01-08',
    #         '2017-01-09',
    #         '2017-01-10']
    # fileList = listdir('groupedby')
    # processedFileList = []
    # for i in range(len(fileList)):
    #     if fileList[i].find('label') > 0:   # 不能单纯用if fileList[i].find('label')，因为结果为-1仍然是代表有值，判断为ture
    #         processedFileList.append(fileList[i])
    # for i in processedFileList:
    #     output_path = "label_2" + i
    filelist = ['22_1', '25_2', '28_4', '34_5', '35_6', '37_3', '39_7']
    for filename in filelist:
        inputpath = 'groupedby\\' + filename + '_label.csv'
        df = pd.read_csv(inputpath, index_col='Time', parse_dates=['Time'])

        outputpath0 = 'label_2\\' + filename + '_8H.csv'
        df_select0 = df['Label'].resample('8H', closed='left', label='left').mean()
        df_select0 = df_select0.fillna(0)
        df_select0.to_csv(outputpath0, mode='a')

        outputpath1 = 'label_2\\' + filename + '_6H.csv'
        df_select1 = df['Label'].resample('6H', closed='left', label='left').mean()
        df_select1 = df_select1.fillna(0)
        df_select1.to_csv(outputpath1, mode='a')

        outputpath2 = 'label_2\\' + filename + '_1H.csv'
        df_select2 = df['Label'].resample('1H', closed='left', label='left').mean()
        df_select2 = df_select2.fillna(0)
        # 由于1月10号的数据不全，所以需要人工进行填充！！！
        df_select2.to_csv(outputpath2, mode='a')

def make_feature():  # 带for循环的
    filelist = ['22_1', '25_2', '28_4', '34_5', '35_6', '37_3', '39_7']
    for filename in filelist:
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
        no_path = 'label\\' + filename + '_result.csv'
        df_no = pd.read_csv(no_path, header= None , sep= '\t')
        no3.extend(df_no.ix[::4, [2]].values.flatten().tolist())
        no2.extend(df_no.ix[1::4, [2]].values.flatten().tolist())
        no0.extend(df_no.ix[2::4, [2]].values.flatten().tolist())
        no1.extend(df_no.ix[3::4, [2]].values.flatten().tolist())

        t8_path = 'label_2\\' + filename + '_8H.csv'
        df_t8 = pd.read_csv(t8_path, header= None)
        t8_0.extend(df_t8.ix[::3, [1]].values.flatten().tolist())
        t8_8.extend(df_t8.ix[1::3, [1]].values.flatten().tolist())
        t8_16.extend(df_t8.ix[2::3, [1]].values.flatten().tolist())

        t6_path = 'label_2\\' + filename + '_6H.csv'
        df_t6 = pd.read_csv(t6_path, header= None)
        t6_0.extend(df_t6.ix[::4, [1]].values.flatten().tolist())
        t6_6.extend(df_t6.ix[1::4, [1]].values.flatten().tolist())
        t6_12.extend(df_t6.ix[2::4, [1]].values.flatten().tolist())
        t6_18.extend(df_t6.ix[3::4, [1]].values.flatten().tolist())

        t1_path = 'label_2\\' + filename + '_1H.csv'
        df_t1 = pd.read_csv(t1_path, header= None)
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
        outputpath = 'label_2\\label_' + filename + '_all.csv'
        df.to_csv(outputpath)

def cluster_all():
    df = pd.read_csv(r'label_2\label_39_7_all.csv')
    df = df.ix[:,1:].fillna(0)
    df_selecet = df.loc[:, :]
    # print df_selecet;exit(0)
    # scaler = StandardScaler().fit(df_selecet)
    # df_pro = scaler.transform(df_selecet)
    #
    # cls = KMeans(n_clusters=4, init='k-means++')  # 分为4类
    # kmeans = cls.fit(df_pro)
    # y = kmeans.predict(df_pro)
    # df['Label'] = y[:]
    # df.to_csv(r'groupedby\39_7_label.csv', index=False)

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

def cluster():
    df = pd.read_csv(r'label_2\label_39_7_all.csv', usecols= ['no0','no1','no2','no3','t6_0','t6_12','t6_18','t6_6','t8_0','t8_16','t8_8'])
    df = df.fillna(0)
    df_selecet = df.loc[:, :]
    # print df_selecet;exit(0)
    # scaler = StandardScaler().fit(df_selecet)
    # df_pro = scaler.transform(df_selecet)
    #
    # cls = KMeans(n_clusters=4, init='k-means++')  # 分为4类
    # kmeans = cls.fit(df_pro)
    # y = kmeans.predict(df_pro)
    # df['Label'] = y[:]
    # df.to_csv(r'groupedby\39_7_label.csv', index=False)

    scaler = StandardScaler().fit(df_selecet)
    df_pro = scaler.transform(df_selecet)

    K = range(1, 11)
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

def time_statistics():
    filelist = ['22_1', '25_2', '28_4', '34_5', '35_6', '37_3', '39_7']
    f = open(r'label_2\time_statistics.csv','a')
    f.write('device'+'\t'+'count_no3' +'\t'+'count_no2'+'\t'+'count_no1'+'\t'+'count_no0')
    f.write('\n')
    for filename in filelist:
        no3 = []
        no2 = []
        no0 = []
        no1 = []
        count_no3 = 0; count_no2 = 0; count_no0 = 0; count_no1 = 0
        inputpath = 'label\\' + filename + '_result.csv'
        df_no = pd.read_csv(inputpath, header=None, sep='\t')
        no3.extend(df_no.ix[::4, [2]].values.flatten().tolist())
        no2.extend(df_no.ix[1::4, [2]].values.flatten().tolist())
        no0.extend(df_no.ix[2::4, [2]].values.flatten().tolist())
        no1.extend(df_no.ix[3::4, [2]].values.flatten().tolist())
        for i in no3:
            count_no3 += i
        for i in no2:
            count_no2 += i
        for i in no1:
            count_no1 += i
        for i in no0:
            count_no0 += i
        line = filename + '\t' + str(count_no3) +'\t'+str(count_no2)+'\t'+str(count_no1)+'\t'+str(count_no0)
        f.write(line)
        f.write('\n')
    f.close()




if __name__ == "__main__":
    # analysis_firstlabel()
    # second_feature()
    # make_feature()
    # cluster_all()
    # cluster()
    time_statistics()