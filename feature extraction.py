import numpy as np
import pandas as pd


cwd2 = "/Users/zhoumanqing/Documents/pycharm/identification/normalplustime.csv"
dataFrame = pd.read_csv(cwd2, delimiter=',',header=None)
dataFrame.columns = ["time","acc-x","acc-y","acc-z"]

#translate the time into index
dataFrame.set_index('time')


dataFrame.rolling(window=10).mean()
print(dataFrame)
#print (dataFrame)

#def sliding_window(data, window_size, step_size):
 #   data = pd.rolling_window(data, window_size)
  #  data = data[step_size - 1 :: step_size]
   # print data
    #return data



#dataFrame['time'] = dataFrame['time'].astype('float64')
#dataFrame['time'] = pd.to_datetime(dataFrame['time'],unit='s')
# Convert column type to be datetime
