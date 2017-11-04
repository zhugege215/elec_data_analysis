#!/user/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import logging

# df = pd.read_csv('a.csv', header=None, nrows=100)
# # print df
# df.to_csv('a_test.csv', header=None)

# 以追加模式创建文件，用来存放切分后的结果，单位：天
f_1216 = open(r'testdata\1216.csv', 'a')
f_1217 = open(r'testdata\1217.csv', 'a')
f_1218 = open(r'testdata\1218.csv', 'a')
f_1219 = open(r'testdata\1219.csv', 'a')
f_1220 = open(r'testdata\1220.csv', 'a')
f_1221 = open(r'testdata\1221.csv', 'a')
f_1222 = open(r'testdata\1222.csv', 'a')
f_1223 = open(r'testdata\1223.csv', 'a')
f_1224 = open(r'testdata\1224.csv', 'a')
f_1225 = open(r'testdata\1225.csv', 'a')
f_1226 = open(r'testdata\1226.csv', 'a')
f_1227 = open(r'testdata\1227.csv', 'a')
f_1228 = open(r'testdata\1228.csv', 'a')
f_1229 = open(r'testdata\1229.csv', 'a')
f_1230 = open(r'testdata\1230.csv', 'a')
f_1231 = open(r'testdata\1231.csv', 'a')
f_0101 = open(r'testdata\0101.csv', 'a')
f_0102 = open(r'testdata\0102.csv', 'a')
f_0103 = open(r'testdata\0103.csv', 'a')
f_0104 = open(r'testdata\0104.csv', 'a')
f_0105 = open(r'testdata\0105.csv', 'a')
f_0106 = open(r'testdata\0106.csv', 'a')
f_0107 = open(r'testdata\0107.csv', 'a')
f_0108 = open(r'testdata\0108.csv', 'a')
f_0109 = open(r'testdata\0109.csv', 'a')
f_0110 = open(r'testdata\0110.csv', 'a')

# 利用字典实现 switch 语法
operation = {
    '2016-12-16':lambda line:f_1216.write(line),
    '2016-12-17':lambda line:f_1217.write(line),
    '2016-12-18':lambda line:f_1218.write(line),
    '2016-12-19':lambda line:f_1219.write(line),
    '2016-12-20':lambda line:f_1220.write(line),
    '2016-12-21':lambda line:f_1221.write(line),
    '2016-12-22':lambda line:f_1222.write(line),
    '2016-12-23':lambda line:f_1223.write(line),
    '2016-12-24':lambda line:f_1224.write(line),
    '2016-12-25':lambda line:f_1225.write(line),
    '2016-12-26':lambda line:f_1226.write(line),
    '2016-12-27':lambda line:f_1227.write(line),
    '2016-12-28':lambda line:f_1228.write(line),
    '2016-12-29':lambda line:f_1229.write(line),
    '2016-12-30':lambda line:f_1230.write(line),
    '2016-12-31':lambda line:f_1231.write(line),
    '2017-01-01':lambda line:f_0101.write(line),
    '2017-01-02':lambda line:f_0102.write(line),
    '2017-01-03':lambda line:f_0103.write(line),
    '2017-01-04':lambda line:f_0104.write(line),
    '2017-01-05':lambda line:f_0105.write(line),
    '2017-01-06':lambda line:f_0106.write(line),
    '2017-01-07':lambda line:f_0107.write(line),
    '2017-01-08':lambda line:f_0108.write(line),
    '2017-01-09':lambda line:f_0109.write(line),
    '2017-01-10':lambda line:f_0110.write(line),
}

def write_file(filename, line):
    operation.get(filename)(line)
i=0
f = open('a.csv', 'r')
for line in f:
    # print line;exit(0)
    i+=1
    col = line.strip().split(",")
    # print type(col[7]);exit()
    # print col[7].split(' ')[0]
    # print col[7].split(' ')[0] == '2016-12-16'
    # f_1216.write(line)
    # exit(0)
    # if col[7].split(' ')[0] == '2016-12-16': # 有重新索引的问题
    #     f_1216.write(line)
    # elif col[7].split(' ')[0] == '2016-12-17':
    #     f_1217.write(line)
    try:
        write_file(col[6].split(' ')[0], line)  # 原始数据只有7列，没有索引
    except IndexError, e:
        print 'error rows = ', i # 一共138453463，最后一行出错，并无大碍
        logging.exception(e)

# 关闭 I/O 资源
f.close()
f_1216.close()
f_1217.close()
f_1218.close()
f_1219.close()
f_1220.close()
f_1221.close()
f_1222.close()
f_1223.close()
f_1224.close()
f_1225.close()
f_1226.close()
f_1227.close()
f_1228.close()
f_1229.close()
f_1230.close()
f_1231.close()
f_0101.close()
f_0102.close()
f_0103.close()
f_0104.close()
f_0105.close()
f_0106.close()
f_0107.close()
f_0108.close()
f_0109.close()
f_0110.close()