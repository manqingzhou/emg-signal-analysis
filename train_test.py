import numpy as np
import pandas as pd
import csv

# user_list="/Users/zhoumanqing/documents/pycharm/user/user1.csv"
# make a empy list, so that we can append stuff inside
user_list = []

for i in range(1,7):
    file_path ='/Users/zhoumanqing/documents/pycharm/user/'+str(i)+'.csv'
    #print(file_path)
    user_list.append((pd.read_csv(file_path,header=None))[0:500])

print(user_list)
    # This needs to be outside the for-loop


with open('/Users/zhoumanqing/documents/pycharm/identification/features.csv','w') as out:
    writer = csv.writer(out,dialect='excel')
    writer.writerow(user_list)
#out.close()
# for i in range(0,len(user_list)):