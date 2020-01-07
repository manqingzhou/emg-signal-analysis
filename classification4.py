#second try decision tree, is is kind of easy and i cannot do for loop
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import math
from statsmodels.tsa.arima_model import ARIMA
from random import gauss
import csv
import ast
df = pd.read_csv("/Users/zhoumanqing/Desktop/data1.csv",sep='\t',header=None)
list = []
for
