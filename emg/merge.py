import pandas as pd
import glob
import numpy as np
from matplotlib import pyplot as plt

path='/Users/zhoumanqing/Documents/pycharm/dataset/S1/S01 Raw Data 30 Files'
all_files = glob.glob(path +'/LiftCycle_0kg*.txt')  #it is a list
list = []
for file in all_files:
    df = pd.read_csv(file, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list.append(df)


dataframe = pd.concat(list,axis=0,ignore_index=True)






