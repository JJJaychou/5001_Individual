# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Load training data
df = pd.read_csv('E:/Study/HKUST/2-Foundation of Data Analytics/Kaggle/train.csv', index_col=0)
#%% see the distribution of label
num_pos = (df['label']==1).sum()
num_neg = (df['label']==0).sum()

#%% Try different models, and select the best model (Random Forest) to generate predictions on test set.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
X = df.dropna(axis = 0).iloc[:,:-1]
y = df.dropna(axis = 0).iloc[:,-1]

clf = RandomForestClassifier(max_depth=4, n_estimators = 500, random_state=5001) # best model!
# clf = svm.SVC(kernel='rbf', C=1, random_state=42)
# clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y, cv=4)
# fit the model
clf.fit(X,y)

#%% Load test data and use the pre-trained model to get the prediction on test set, and then save the prediction to the target file.
df_test = pd.read_csv('E:/Study/HKUST/2-Foundation of Data Analytics/Kaggle/test.csv', index_col=0)
pred = clf.predict(df_test)
df_pred = pd.DataFrame({'id': df_test.index, 'label':pred})
df_pred.to_csv('E:/Study/HKUST/2-Foundation of Data Analytics/Kaggle/submission_final.csv', index = False)
