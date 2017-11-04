#!/user/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np

pd.set_option('display.width', 200)
names = ['DataId', 'NodeAddr', 'DevAddr', 'DevChannel', 'Data', 'SampleType', 'Time']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# 先观察值
# day_data = pd.read_csv(r'testdata\1221.csv', names=names)
# print day_data.NodeAddr.value_counts()
'''
37    758025    空3
28    602748    空4
35    595107    空6
39    582885    空7
25    567099    空2 
24    552402
27    532728
22    438822    空1
32    419625
34    278658    空5
'''

# day_data = pd.read_csv(r'testdata\0108.csv', names=names)
# print day_data.NodeAddr.value_counts()
'''
37    710316    空3
35    626580    空6
28    614637    空4
39    581994    空7
27    493299
24    391131
22    317466    空1
25    307971    空2
32    259605
34    217818    空5
'''

print 'start...'
def splitFile():
    i = 0
    f_22_1 = open(r'a_split_bydevice_notmean\22_1.csv', 'a')
    f_25_2 = open(r'a_split_bydevice_notmean\25_2.csv', 'a')
    f_37_3 = open(r'a_split_bydevice_notmean\37_3.csv', 'a')
    f_28_4 = open(r'a_split_bydevice_notmean\28_4.csv', 'a')
    f_34_5 = open(r'a_split_bydevice_notmean\34_5.csv', 'a')
    f_35_6 = open(r'a_split_bydevice_notmean\35_6.csv', 'a')
    f_39_7 = open(r'a_split_bydevice_notmean\39_7.csv', 'a')

    operation = {
        '0022': lambda line: f_22_1.write(line),
        '0025': lambda line: f_25_2.write(line),
        '0037': lambda line: f_37_3.write(line),
        '0028': lambda line: f_28_4.write(line),
        '0034': lambda line: f_34_5.write(line),
        '0035': lambda line: f_35_6.write(line),
        '0039': lambda line: f_39_7.write(line),
    }

    with open(r'a.csv','r') as fs:
        for line in fs:
            col = line.strip().split(',')
            try:
                operation.get(str(col[1]), lambda x: None)(line)
            except Exception, e:
                print Exception, ':', e
                print 'error rows = ', i
            i += 1

    f_22_1.close()
    f_25_2.close()
    f_37_3.close()
    f_28_4.close()
    f_34_5.close()
    f_35_6.close()
    f_39_7.close()

if __name__ == "__main__":
    splitFile()