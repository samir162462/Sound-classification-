# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:38:54 2020

@author: sam
"""

import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

import IPython.display as ipd
import sounddevice as sd
import time


def return_audio_in_time(audio,sr,time_start,time_end):
       
        t1 =time_start*sr
        t1 = int(t1)
        t2=time_end*sr
        t2 = int(t2)
        audio = audio[t1:t2]
        print(librosa.get_duration(y=audio,sr=sr))
        return audio
        
for i in range(1+10,10+10):
    x, sr = librosa.load('train_test_50_50/test_y_mr/New_MR_0'+str(i)+'.wav')
    
    print(x.shape)
    
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)\
    
    
    onset_frames = librosa.onset.onset_detect(x, sr=sr)
    print(onset_frames) # frame numbers of estimated onsets
    
    
    onset_times = librosa.frames_to_time(onset_frames)
    print(onset_times)
    
    
    S = librosa.stft(x)
    logS = librosa.amplitude_to_db(abs(S))
    librosa.display.waveplot(logS-np.mean(logS), sr=sr)\
    
    clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(x), hop_length=512)
    print(np.mean(clicks))
    xs =  x + clicks
    print(x.dtype)
    print(xs.dtype)
    
    
    
    
    
    
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x,sr=sr)
    lims = plt.gca().get_ylim()
    plt.vlines(onset_times, lims[0], lims[1], color='lime', alpha=0.9,
               linewidth=2, label='Beats')
    
    plt.legend(frameon=True, shadow=True)
    plt.title('CQT + Beat and sub-beat markers in AS sound')
    plt.tight_layout()
    #plt.savefig('CQT + Beat and sub-beat markers in AS Sound 170')
    plt.show()
    sd.play(xs, sr)
    time.sleep(2.4)





