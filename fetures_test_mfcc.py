# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:11:10 2019

@author: samir filfil
"""
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import numpy as np
spctro_dis =1
import os
import IPython
import time
from datetime import timedelta as td
import librosa
from scipy import signal
from matplotlib import pyplot as plt
import sounddevice as sd
import os
import pywt
import pywt.data
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile
from scipy.fftpack import fft, rfft, irfft



X, sample_rate = librosa.load("Trace/AS_001.wav")
stft = np.abs(librosa.stft(X))
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)    
plt.plot(X)

fft_f = fft(X)
fft_sf = np.sort(sam)[::-1]
fft_fn =sam[0:80]

arrays_sound = np.empty((0,193))
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
tonnets = np.mean(librosa.feature.tempogram(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)

display(ss.shape)


plt.plot(arrays_sound)


c = 0
for f in ss:
        display(f)
        c=c+1
display(c)

