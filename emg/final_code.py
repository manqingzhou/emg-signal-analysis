import pandas as pd
import glob
import numpy as np
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#make five empty list for store different loads
list0kg,list2pt5kg,list5kg,list10kg,list15kg,list20kg=([] for i in range(6))

#put all the 30 raw file into one single list and then slide them into each empty list
#create an empty list first
#the glob.glob function is for connecting the file, its the path/LiftCyvle_0kg1.txt,* match everything
path='/Users/zhoumanqing/Documents/pycharm/dataset/S1/S01 Raw Data 30 Files'
weight=['0','2pt5','5','10','15','20']
list = []
for x in weight:
    all_files = glob.glob(path + '/LiftCycle_' + x + '*.txt')
    for file in all_files:
        list.append(file)
print(list)

#slide the signal into different dataframes
for a in list[0:5]:
    df = pd.read_csv(a, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list0kg.append(df)
    list0kg = pd.concat(list0kg, axis=0, ignore_index=True)
print(list0kg)



for b in list[5:10]:
    df = pd.read_csv(b, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list2pt5kg.append(df)
    data2pt5kg = pd.concat(list2pt5kg, axis=0, ignore_index=True)



for c in list[10:15]:
    df = pd.read_csv(c, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list5kg.append(df)
    data5kg = pd.concat(list5kg, axis=0, ignore_index=True)

for d in list[15:20]:
    df = pd.read_csv(d, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list10kg.append(df)
    data10kg= pd.concat(list10kg, axis=0, ignore_index=True)




for e in list[20:25]:
    df = pd.read_csv(e, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list15kg.append(df)
    data15kg = pd.concat(list15kg, axis=0, ignore_index=True)


for f in list[25:30]:
    df = pd.read_csv(f, delimiter="\t", header=None,
                     names=['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'Handswitch', 'boxswitch', 'MS1', 'MS2', 'MS3', 'MS4',
                            'MS5', 'MS6'])
    list20kg.append(df)
    data20kg = pd.concat(list20kg, axis=0, ignore_index=True)



#normalized function, mq is the array of the array
def normalizaiton(data):
    columns = ['BB', 'TB', 'BR', 'AD', 'LES', 'TES']
    mq = []
    for x in columns:
        df = pd.DataFrame(data, columns=np.array(x))
        df2 = np.array(df).ravel()
        # change it into 1D array to do window sliding
        # np.sqrt(sum([a[window_size-i-1:len(a)-i]**2 for i in range(window_size-1)])/window_size)
        df3 = np.sqrt(sum([df2[100 - i - 1:len(df2) - i] ** 2 for i in range(99)]) / 100)
        max_rms = 1.0608
        normalized = np.divide(df3, max_rms)
        mq.append(normalized)
    return mq

nordata0kg=normalizaiton(data0kg)

nordata2pt5kg=normalizaiton(data2pt5kg)
nordata5kg=normalizaiton(data5kg)
nordata10kg=normalizaiton(data10kg)
nordata15kg=normalizaiton(data15kg)
nordata20kg=normalizaiton(data20kg)

#this is for get the list(each muscle)out of the data and then convert them into dataframe and merge them together
#dont forget the axis=1 to make sure each array is each column instead of each array becomes each row
def merge(data):
    BB = pd.DataFrame(data[0], columns=['BB'])
    TB = pd.DataFrame(data[1], columns=['TB'])
    BR = pd.DataFrame(data[2], columns=['BR'])
    AD = pd.DataFrame(data[3], columns=['AD'])
    LES = pd.DataFrame(data[4], columns=['LES'])
    TES = pd.DataFrame(data[5], columns=['TES'])
    normerge= pd.concat([BB, TB, BR, AD, LES, TES], axis=1)
    return normerge

merge0kg=merge(nordata0kg)
merge2pt5kg=merge(nordata2pt5kg)
merge5kg=merge(nordata5kg)
merge10kg=merge(nordata10kg)
merge15kg=merge(nordata15kg)
merge20kg=merge(nordata20kg)


#this is for the labeling
idx = 0 #first column
merge0kg.insert(loc=idx, column='label',value=0)
merge0kg.label=0
merge2pt5kg.insert(loc=idx, column='label',value=0)
merge2pt5kg.label=1
merge5kg.insert(loc=idx, column='label',value=0)
merge5kg.label=2
merge10kg.insert(loc=idx, column='label',value=0)
merge10kg.label=3
merge15kg.insert(loc=idx, column='label',value=0)
merge15kg.label=4
merge20kg.insert(loc=idx, column='label',value=0)
merge20kg.label=5

merge_all=pd.concat([merge0kg,merge2pt5kg,merge5kg,merge10kg,merge15kg,merge20kg])

X = merge_all.drop('label',axis=1)
y = merge_all ['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)   #training = 0.8/testing =0.2
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

regressor= RandomForestClassifier(n_estimators=500,max_depth=20,random_state=42)
regressor.fit(X_train,y_train)

#make prediction
y_pred = regressor.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#this is for the test of the module
#neighbors=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51]
#cv_scores= []

#for k in neighbors:
    #knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(X_train, y_train)
    #scores = cross_val_score(knn,X_train,y_train,cv =10,scoring='accuracy')
    #cv_scores.append(scores.mean())

#MSE = [1 - x for x in cv_scores]
#optimal_k =neighbors[MSE.index(min(MSE))]

#plt.plot(neighbors,MSE)
#plt.xlabel('Number of Neighbors K')
#plt.ylabel('Misclassification Error')
#plt.show()



















