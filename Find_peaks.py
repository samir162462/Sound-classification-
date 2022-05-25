# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:41:32 2020

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

import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks



def return_audio_in_time(audio,sr,time_start,time_end):
       
        t1 =time_start*sr
        t1 = int(t1)
        t2=time_end*sr
        t2 = int(t2)
        audio = audio[t1:t2]
        print(librosa.get_duration(y=audio,sr=sr))
        return audio
        
        
def set_peaks(file):
        y, sr = librosa.load(file)
        
        x = y
        peaks, properties = find_peaks(x, prominence=1, width=80)
        print(peaks)
        properties["prominences"], properties["widths"]
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
                   ymax = x[peaks], color = "C1")
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                   xmax=properties["right_ips"], color = "C1")
        plt.show()
        
        
        s_arr = np.zeros(len(peaks)-1)
        pointer1 = 0 
        pointer2 = 0 
        counter = 0 
        
        s=0
        pointer1 = peaks[0]
        pointer2 = peaks[1] 
        for i in range(len(peaks)-1):
                
                if pointer2-pointer1>1000:
                        s_arr[counter] = pointer1
                        counter +=1
                        s_arr[counter] = pointer2
                        counter +=1
                        pointer1 = pointer2
                        pointer2 = peaks[i]
                        i=i+1
                        
                else:
                        
                        pointer2 = peaks[i]
                        print(s_arr)
        s_arr.astype(int)
            
        uniques = np.unique(s_arr)
        print(uniques[1])
        shift = 0
        for i in range(len(uniques)-1):
                
                if uniques[i+1]-uniques[i]<1000:
                        continue
                else:
                        print(i)
        
                        plt.plot(y[int(uniques[i]):int(uniques[i+1])])
                        plt.show()

set_peaks('train_test_50_50/train_y_mr/')