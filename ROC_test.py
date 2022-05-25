# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:34:52 2020

@author: samir filfil
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier




def roc_test_plot(y_test,y_score,n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true = y_test[i*100:(1+i)*100,1], y_score = y_score[i*100:(1+i)*100,1],pos_label = i)
#        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
    # Compute micro-average ROC curve and ROC area
#    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    # Plot ROC curve
    plt.figure()
#    plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    
    
    
    
# Import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#
## Binarize the output
#y = label_binarize(y, classes=[0, 1, 2])
#n_classes = y.shape[1]
#
## Add noisy features to make the problem harder
#random_state = np.random.RandomState(0)
#n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#
## shuffle and split training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
#                                                    random_state=0)
## Learn to predict each class against the other
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                 random_state=random_state))
#y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#print(y_score)
#print(y_test)
## Compute ROC curve and ROC area for each class
#roc_test_plot(y_test,y_score,n_classes)