# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 03:54:55 2020

@author: sam
"""

import librosa
import matplotlib.pyplot as plt
import os 
from scipy.signal.signaltools import wiener
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from playsound import playsound

#N1, SR = librosa.load('train_test_50_50 - random/as/New_AS_001.wav') # Downsample 44.1kHz to 8kHz
#plt.plot(N1)

file_holder = []
listOfFiles = os.listdir('predict_folder')
    
for file in listOfFiles:
    filename = os.path.join('predict_folder',file)
    file_holder.append(filename)
print(file_holder)

for i in file_holder:
    N, SR = librosa.load(i)
   
    librosa.output.write_wav(i, N[50000:100000], 8000)
