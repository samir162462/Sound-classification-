# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:45:17 2020

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
from pydub import AudioSegment


#y, sr = librosa.load('upnormal/train_y_as/New_AS_120.wav')

def return_audio_in_time(audio,sr,time_start,time_end):
       
        t1 =time_start*sr
        t1 = int(t1)
        t2=time_end*sr
        t2 = int(t2)
        audio = audio[t1:t2]
        print(librosa.get_duration(y=audio,sr=sr))
        return audio
        
        
        

y, sr = librosa.load('upnormal/test_y_As/New_AS_148.wav')
print(return_audio_in_time(y,sr,0,2))
#y = return_audio_in_time(y,sr, 0.06965986,0.11609977)
print(sr)
#y, sr = librosa.load('test_x/New_N_131.wav')
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512))
subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)
print(beat_times)
print(subseg)
print(subseg_t)


plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(cqt,
                                                 ref=np.max),
                         y_axis='cqt_hz', x_axis='time')
lims = plt.gca().get_ylim()
plt.vlines(beat_times, lims[0], lims[1], color='lime', alpha=0.9,
           linewidth=2, label='Beats')
plt.vlines(subseg_t, lims[0], lims[1], color='linen', linestyle='--',
           linewidth=1.5, alpha=0.5, label='Sub-beats')
plt.legend(frameon=True, shadow=True)
plt.title('CQT + Beat and sub-beat markers in AS sound')
plt.tight_layout()
plt.savefig('CQT + Beat and sub-beat markers in AS Sound 170')
plt.show()