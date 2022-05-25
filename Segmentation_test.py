# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 02:05:20 2020

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
from sklearn.metrics import classification_report
from sklearn import tree
import matplotlib.patches as mpatches
import librosa.display



y, sr = librosa.load('upnormal/train_y_as/New_AS_001.wav')
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
bounds = librosa.segment.agglomerative(chroma, 20)
bound_times = librosa.frames_to_time(bounds, sr=sr)
bound_times


plt.figure()
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--',
            linewidth=2, alpha=0.9, label='Segment AS boundaries')
plt.axis('tight')
plt.legend(frameon=True, shadow=True)
plt.title('Power spectrogram')
plt.tight_layout()
plt.savefig('segment_normal_AS_01.png')
plt.show()
print('done')