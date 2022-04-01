# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:42:41 2022

@author: MXR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv('E:/Study/HKUST/2-Foundation of Data Analytics/Kaggle/train.csv', index_col=0)


#%%
num_pos = (df['label']==1).sum()
num_neg = (df['label']==0).sum()

#%%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
X = df.dropna(axis = 0).iloc[:,:-1]
y = df.dropna(axis = 0).iloc[:,-1]

clf = RandomForestClassifier(max_depth=4, n_estimators = 300, random_state=0)
# clf = svm.SVC(kernel='linear', C=1, random_state=42)
# clf = AdaBoostClassifier(n_estimators=300)
scores = cross_val_score(clf, X, y, cv=6)
scores
#%%
clf.fit(X,y)
#%%
df_test = pd.read_csv('E:/Study/HKUST/2-Foundation of Data Analytics/Kaggle/test.csv', index_col=0)
pred = clf.predict(df_test)
df_pred = pd.DataFrame({'id': df_test.index, 'label':pred})
df_pred.to_csv('pred1.csv', index = False)
#%%
df_pos = df[df['label']==1]
df_neg = df[df['label']==0]

train_pos = df_pos.sample(6000, random_state = 0, axis = 0)
test_pos = df_pos[~df_pos.index.isin(train_pos.index)]

test_neg = df_neg.sample(1506, random_state = 0, axis = 0)
train_neg = df_neg[~df_neg.index.isin(test_neg.index)]

#%%
print(len(train_neg)/len(test_neg))

train_pos_upsample = train_pos
for i in range(50):
    train_pos_upsample = pd.concat([train_pos_upsample, train_pos], axis = 0)

train_neg_downsample = train_neg.sample(len(train_pos_upsample), random_state = 0, axis = 0)

#%%
train_X = pd.concat([train_pos_upsample, train_neg_downsample], axis = 0)
train_y = train_X['is_fraud']
del train_X['is_fraud']

test_X = pd.concat([test_pos, test_neg], axis = 0)
test_y = test_X['is_fraud']
del test_X['is_fraud']

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(train_X, train_y)

y_pred = clf.predict(test_X)

from sklearn.metrics import classification_report

target_names = ['not_fraud', 'is_fraud']
print(classification_report(test_y, y_pred, target_names=target_names))
