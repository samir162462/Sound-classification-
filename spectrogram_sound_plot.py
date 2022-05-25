# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:34:26 2020

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


spctro_dis =1

if spctro_dis ==1:
    ts = os.listdir("test")
    fig = plt.figure(figsize=(10, 6), dpi=300)
    for i in ts:
       
        Audiodata, sample_rate = librosa.load("test/"+i)        
        N = 512 #Number of point in the fft
        f, t, Sxx = signal.spectrogram(Audiodata, sample_rate,window = signal.blackman(N),nfft=N)
       
        
        plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
        #plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [seg]')
        plt.title(i,size=16);
        
        plt.show()
        fig.savefig(i+"sounds.png", dpi = 300) # save figure

