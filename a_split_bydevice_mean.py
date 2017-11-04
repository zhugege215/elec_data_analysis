#!/user/bin/env python
# -*- coding: utf-8 -*-
# 对 a_split_bydevice_notmean 的结果先去掉功率数据，之后进行数据变换，最后进行均值化并且去噪

from os import listdir
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

names = ['DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Data', 'SampleType', 'Time']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def dataTransform():
    outputfileList = listdir('a_split_bydevice_mean')

    if len(outputfileList) == 0:
        print 'dir is None, start to creat file in which the power will be removed... \nthen you should ' \
              'run this program again!'

        inputfileList = listdir('a_split_bydevice_notmean')
        m = len(inputfileList)
        for i in range(m):
            count = 0
            fileName = inputfileList[i]
            inputfilePath = 'a_split_bydevice_notmean\\' + fileName
            outputfilePath = 'a_split_bydevice_mean\\' + 'removed_' + fileName
            outputfile = open(outputfilePath, 'a')
            with open(inputfilePath, 'r') as fs:
                for line in fs:
                    count += 1
                    try:
                        col = line.strip().split(',')
                        if col[5] == '2':
                            pass
                        else:
                            outputfile.write(line)
                    except Exception, e:
                        print Exception, ':', e
                        print 'error rows = ', count
            outputfile.close()
    else:
        m = len(outputfileList)
        for i in range(m):
            fileName = outputfileList[i]
            inputfilePath = 'a_split_bydevice_mean\\' + fileName
            outputfilePath = 'a_split_bydevice_mean\\' + 'transformed_' + fileName
            df = pd.read_csv(inputfilePath, names=names)
            df_cur = df[df.SampleType == 0].copy()
            df_vol = df[df.SampleType == 1].copy()

            df_cur.insert(5, 'Vol', df_vol['Data'].values)
            df_cur.rename(columns={'Data': 'Cur'}, inplace=True)
            del df_cur['SampleType']
            df_cur.to_csv(outputfilePath, index=False)

def equalization():
    fileList = listdir('a_split_bydevice_mean')
    processedFileList = []
    for i in range(len(fileList)):
        if fileList[i].find("transformed") == 0:
            processedFileList.append(fileList[i])
    for i in range(len(processedFileList)):
        inputFile = processedFileList[i]
        inputFilePath = 'a_split_bydevice_mean\\' + inputFile
        df = pd.read_csv(inputFilePath)
        filt_mean = df.iloc[1::3, :].copy()
        curdata = df['Cur'].values
        voldata = df['Vol'].values
        N = len(curdata)
        a = np.arange(0,N)
        cur_mean = []
        vol_mean = []
        for j in a[1::3]:
            b = (curdata[j - 1] + curdata[j] + curdata[j + 1]) / 3
            cur_mean.append(b)
        filt_mean['Cur'] = cur_mean[:]
        for jj in a[1::3]:
            b = (voldata[jj - 1] + voldata[jj] + voldata[jj + 1]) / 3
            vol_mean.append(b)
        filt_mean['Vol'] = vol_mean[:]

        #只对电流数据进行去噪
        xx = filt_mean['Cur'].values
        width = 50
        delta = 10
        eps = 1.5
        NN = len(xx)
        abnormal = []
        for k in np.arange(0, NN - width, delta):
            s = xx[k:k + width]
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
        br.fit(t[select].reshape(-1, 1), xx[select])
        y = br.predict(np.arange(NN).reshape(-1, 1))
        y[select] = xx[select]
        filt_mean['Cur'] = y[:]

        outputFilePath = r'a_split_bydevice_mean\mean_denoising_'+inputFile
        filt_mean.to_csv(outputFilePath, index=False)


if __name__ == "__main__":
    # dataTransform()
    equalization()


