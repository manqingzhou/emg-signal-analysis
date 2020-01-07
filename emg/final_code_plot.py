import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

#path of the file(for the part one, we need two subjects for three loads) and read the txt file, the \t is tab and the name is for the column
cwd = "/Users/zhoumanqing/Documents/pycharm/dataset/S2/S02 Raw Data 30 Files/LiftCycle_2pt5kg1.txt"
Dataframe=pd.read_csv(cwd,delimiter="\t", names=['BB','TB','BR','AD','LES','TES','Handswitch','boxswitch','MS1','MS2','MS3','MS4','MS5','MS6'])
#choose the muscle we need to plot in the question 1.
columns=['BB','TB','BR','AD','LES','TES','boxswitch']

#seperate the file into single column for the future filter
dfBB = pd.DataFrame(Dataframe, columns=['BB'])
dfTB = pd.DataFrame(Dataframe, columns=['TB'])
dfBR = pd.DataFrame(Dataframe, columns=['BR'])
dfAD = pd.DataFrame(Dataframe, columns=['AD'])
dfLES = pd.DataFrame(Dataframe, columns=['LES'])
dfTES = pd.DataFrame(Dataframe, columns=['TES'])
dfbox =pd.DataFrame(Dataframe,columns=['boxswitch'])

#the function for the high-pass filter, this function only work for single array, so we use np.ravel to covert the dataframe to the array
#we set the cutoff 20 and the order 4
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high')
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, np.ravel(data))
    return y

#filtered_xx is the dataframe after the high-pass
filtered_BB=pd.DataFrame(butter_highpass_filter(dfBB,20,1000),columns=['BB'])
filtered_TB=pd.DataFrame(butter_highpass_filter(dfTB,20,1000),columns=['TB'])
filtered_BR=pd.DataFrame(butter_highpass_filter(dfBR,20,1000),columns=['BR'])
filtered_AD=pd.DataFrame(butter_highpass_filter(dfAD,20,1000),columns=['AD'])
filtered_LES=pd.DataFrame(butter_highpass_filter(dfLES,20,1000),columns=['LES'])
filtered_TES=pd.DataFrame(butter_highpass_filter(dfTES,20,1000),columns=['TES'])
filtered_all=pd.concat([filtered_BB,filtered_TB,filtered_BR,filtered_AD,filtered_LES,filtered_TES],axis=1)


#smooth the signal with window rms, the window size is 100 and the important thing about this part is that, make sure df2 is array
#we calculate the max-rms of the whole load cuz we need to normalize the signal by sub-maximal normalization
columns=['BB','TB','BR','AD','LES','TES']
for x in columns:
    df = pd.DataFrame(filtered_all, columns=np.array(x))
    df2 = np.array(df).ravel()
    #change it into 1D array to do window sliding
    # np.sqrt(sum([a[window_size-i-1:len(a)-i]**2 for i in range(window_size-1)])/window_size)
    df3 = np.sqrt(sum([df2[100-i-1:len(df2)-i]**2 for i in range(99)])/100)
    max_rms = 1.0608
    normalized=np.divide(df3,max_rms)
    rms=np.sqrt(np.mean(normalized**2))
    print(rms)

#this is for calculate the whole signal amplitude for the part one question
#for x in columns:
    #df=pd.DataFrame(filtered_all,columns=np.array(x))
    #dn = np.power(df, 2)
    #rms = np.sqrt(np.mean(dn))
    #print(rms)

#function for normalization,
def rms(data):
    df = pd.DataFrame(data)
    df2 = np.array(df).ravel()
    # change it into 1D array to do window sliding
    # np.sqrt(sum([a[window_size-i-1:len(a)-i]**2 for i in range(window_size-1)])/window_size)
    df3 = np.sqrt(sum([df2[100 - i - 1:len(df2) - i] ** 2 for i in range(99)]) / 100)
    #max_rms = 1.0608
    #normalized = np.divide(df3, max_rms)
    return df3

def rms(data):
    df=np.array(data).ravel()
    df2=np.sqrt(sum([df[100 - i - 1:len(df) - i] ** 2 for i in range(99)]) / 100)
    return df2

#plot the normalized the signal with all the muscle and seperated box_switch plot.
nordataBB=pd.DataFrame(rms(filtered_BB),columns=['BB'])
nordataTB=pd.DataFrame(rms(filtered_TB),columns=['TB'])
nordataBR=pd.DataFrame(rms(filtered_BR),columns=['BR'])
nordataAD=pd.DataFrame(rms(filtered_AD),columns=['AD'])
nordataLES=pd.DataFrame(rms(filtered_LES),columns=['LES'])
nordataTES=pd.DataFrame(rms(filtered_TES),columns=['TES'])
#for a0,a1,a2,a3,a4 and a4 are name for the subplot,the label is for identifying the plot
time = np.array([i / 1000 for i in range(0, len(nordataLES), 1)])
time2 = np.array([i / 1000 for i in range(0, len(Dataframe), 1)])
f, (a0, a1,a2,a3,a4,a5) = plt.subplots(nrows=6,ncols=1,figsize=(7,7))
a0.plot(time, nordataBB,label='BBmuscle')
a0.legend()
a0.set_title('amplitude estimation EMG signal for subject2')
a1.plot(time, nordataTB,label='TBmuscle')
a1.legend()
a2.plot(time,nordataBR,label='BRmuscle')
a2.legend()
a3.plot(time, nordataAD,label='ADmuscle')
a3.legend()
a4.plot(time, nordataLES,label='LESmuscle')
a4.legend()
a5.plot(time, nordataTES,label='TESmuscle')
a5.legend()
a5.set_xlabel('time(second)')
f.tight_layout()
plt.savefig("subject2_3")









