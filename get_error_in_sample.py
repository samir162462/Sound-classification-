
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 02:51:52 2020

@author: samir filfil
"""
from numpy import savetxt
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
from pandas import DataFrame
from numpy import loadtxt
import matplotlib as mpl 
import pandas as pd

i=1


for m in range(30):
        try:
                
                full_sound_feature = loadtxt('beat feature/beat'+str(m)+'.csv',dtype=complex, delimiter=',')
        except:
                continue
        def get_mean(x):
                z = np.zeros(1)
                for i in range(200):
                        z = z + x[i]
                return z 
                        
        sounds_array = np.empty((5,200))
        
        mean1 = np.sum(full_sound_feature[0:200][:])/200
        mean2 = np.sum(full_sound_feature[200:400][:])/200
        mean3 = np.sum(full_sound_feature[400:600][:])/200
        mean4 = np.sum(full_sound_feature[600:800][:])/200
        mean5 = np.sum(full_sound_feature[800:1000][:])/200
        
        std1 = np.std(full_sound_feature[0:200][:])
        std2 = np.std(full_sound_feature[200:400][:])
        std3 = np.std(full_sound_feature[400:600][:])
        std4 = np.std(full_sound_feature[600:800][:])
        std5 = np.std(full_sound_feature[800:1000][:])
        
        print([mean1,mean2,mean3,mean4,mean5])
        print([std1,std2,std3,std4,std5])
        
        
        
        labels = ['Normal', 'MR', 'MVP','AS','MS']
        x_pos = np.arange(len(labels))
        CTEs = [mean1, mean2, mean3,mean4,mean5]
        error = [std1, std2, std3,std4,std5]
        
        # Create a figure with customized size

        
        
        # Set the axis lables
        
        
        
        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs,
               yerr=error,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)
        ax.set_xlabel('Type', fontsize = 18)
        ax.set_ylabel('Value', fontsize = 18)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title('ERR CQT '+str(m)+'')
        ax.yaxis.grid(True)
        
        # Save the figure and show
        plt.tight_layout()
        plt.savefig('Error in samples/Err_CQT'+str(m)+'.png')
        plt.show()
