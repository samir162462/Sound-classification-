# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 02:42:50 2020

@author: samir filfil
"""
import os
import librosa
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy import signal
import pickle
import heapq
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.patches as mpatches

def createList(r1, r2): 
    return [item for item in range(r1, r2+1,2)] 

i=1
fig = plt.figure()
plt.figure()
ax = fig.add_subplot(111)
one = mpatches.Patch(facecolor='red', label='Normal', linewidth = 0.5)
two = mpatches.Patch(facecolor='blue', label = 'MR', linewidth = 0.5)
three = mpatches.Patch(facecolor='grey', label = 'MVP', linewidth = 0.5)
four = mpatches.Patch(facecolor='green', label = 'AS', linewidth = 0.5)
five = mpatches.Patch(facecolor='orange', label = 'MS', linewidth = 0.5)

legend = plt.legend(handles=[one, two, three,four,five], loc = 4, fontsize = 'small')

x = np.random.random_integers(0, 200, size=(200,200))
ax = plt.gca()
for i in range(200 ):
        #ax.plot(x[i], x[i]+i,'go')
        ax.set_xticks(createList(0, 200))
        ax.boxplot(x[i], x[i]+i,'go')
plt.savefig('mfccs')
