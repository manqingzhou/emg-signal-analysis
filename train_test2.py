import numpy as np
import pandas as pd
import csv


fout = open("/Users/zhoumanqing/documents/pycharm/identification/features.csv","a")
for line in open("/Users/zhoumanqing/documents/pycharm/user/1.csv"):
    fout.write(line)
for num in range(2,7):
    f=open('/Users/zhoumanqing/documents/pycharm/user/'+str(num)+'.csv')
    f.__next__()
    for line in f:
        fout.write(line)
