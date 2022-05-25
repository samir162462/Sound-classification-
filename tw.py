# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:41:41 2020

@author: sam
"""

from smart_a_data import smart_adaptive_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


iris = datasets.load_iris()
X = iris.data
y = iris.target

target_names = ['Flower 1 ', 'Flower 2','Flower 3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

print(X_train.shape)
print(X[75:150].shape)
#
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
pre = clf.predict(X_test)
print(pre)
#print(len([[2,3,4,5],[4,5,3,2]]))
#print(y[0:75])
classfication_model = classification_report(y_test,pre, labels=[0, 1,2])
print(classfication_model)


x_tr = [] 
yy = [] 

x1,y1 =  smart_adaptive_dataset(X[0:25],X[25:50], epoch= 3,AI_active= False,min_max= True,work_on = '0',type_d='mean',samples_count=50)
x2,y2 =  smart_adaptive_dataset(X[50:75],X[75:100], epoch= 3,AI_active= False,min_max= True,work_on = '0',type_d='mean',samples_count=50)
x3,y3 =  smart_adaptive_dataset(X[100:125],X[125:150], epoch= 3,AI_active= False,min_max= True,work_on = '0',type_d='mean',samples_count=50)

x_tr.append(x1)
x_tr.append(x2)
x_tr.append(x3)
x_tr = np.array(x_tr).flatten()

yy.append(y1)
yy.append(y2)
yy.append(y3)
yy = np.array(yy).flatten()

clf = DecisionTreeClassifier()
clf = clf.fit(x_tr,np.array(y[0:25],y[50:50],y[0:75]).flatten())
pre = clf.predict(ss)
print(pre)
print()
print(y[75:150])
#classfication_model = classification_report(y[75:150],pre, labels=[0, 1,2],target_names=target_names )