import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from scipy import ndimage
import string
import sys

cwd = "/Users/zhoumanqing/Documents/pycharm/dataset/S1/S01 Raw Data 30 Files/LiftCycle_20kg1.txt"
data=pd.read_csv(cwd,delimiter="\t", names=['BB','TB','BR','AD','LES','TES','Handswitch','boxswitch','MS1','MS2','MS3','MS4','MS5','MS6'])


columns=['BB','TB','BR','AD','LES','TES','boxswitch']
    #df2=np.array(df).ravel()  #change it into 1D array to do window sliding
    # np.sqrt(sum([a[window_size-i-1:len(a)-i]**2 for i in range(window_size-1)])/window_size)
    #df3=np.sqrt(sum([df2[100-i-1:len(df2)-i]**2 for i in range(99)])/100)
    #df4=pd.DataFrame(df3)
    #dn = np.power(df, 2)
    #rms = np.sqrt(np.mean(dn))
    #print(rms)
time = np.array([i / 1000 for i in range(0, len(data), 1)])
dfBB = pd.DataFrame(data, columns=['BB'])
plt.subplot(7, 1, 1)
plt.plot(time, dfBB)
plt.xlabel('time(sec)')
plt.title('BBmuscle')
dfTB = pd.DataFrame(data, columns=['TB'])
plt.subplot(7, 1, 2)
plt.plot(time, dfTB,label='TBmuscle')
plt.xlabel('time(sec)')
plt.title('TBmuscle')
dfBR = pd.DataFrame(data, columns=['BR'])
plt.subplot(7, 1, 3)
plt.plot(time, dfBR)
plt.xlabel('time(sec)')
plt.title('BRmuscle')
dfAD = pd.DataFrame(data, columns=['AD'])
plt.subplot(7, 1, 4)
plt.plot(time, dfAD)
plt.xlabel('time(sec)')
plt.title('ADmuscle')
dfLES = pd.DataFrame(data, columns=['LES'])
plt.subplot(7, 1, 5)
plt.plot(time, dfLES)
plt.xlabel('time(sec)')
plt.title('LESmuscle')
dfTES = pd.DataFrame(data, columns=['TES'])
plt.subplot(7, 1, 6)
plt.plot(time, dfTES)
plt.xlabel('time(sec)')
plt.title('TESmuscle')
plt.tight_layout()
plt.show()









#print (df)
#print(df.mean())



#dn=np.power(df,2)
#rms= np.sqrt(np.mean(dn))
#print(rms)
#df.rolling(500).mean().plot()
#plt.show()

#dn=df.values

#df2 = dn[0:99]
#df3 = dn[100:199]
#df4 = dn[200:299]













