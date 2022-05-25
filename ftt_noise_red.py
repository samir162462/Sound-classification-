# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:45:25 2019

@author: samir filfil
"""
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

import numpy as np
from scipy.io import wavfile

AudioName = "extra_normal_s1s2/"+'s1s2normal__106_1306776721273_B1.wav'
fs, Audiodata = wavfile.read(AudioName)
X, sample_rate = librosa.load(AudioName)

# Plot the audio signal in time
plt.plot(Audiodata)
plt.title('Audio signal in time',size=16)

# spectrum
from scipy.fftpack import fft # fourier transform
n = len(X) 
AudioFreq = fft(X)
AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
MagFreq = np.abs(AudioFreq) # Magnitude
MagFreq = MagFreq / float(n)
# power spectrum
MagFreq = MagFreq**2
if n % 2 > 0: # ffte odd 
    MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
else:# fft even
    MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2 

plt.figure()
freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (sample_rate / n);
plt.plot(fft(X)) #Power spectrum
plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq)) #Power spectrum
plt.xlabel('Frequency (kHz)'); plt.ylabel('Power spectrum (dB)');


#Spectrogram
N = 512 #Number of point in the fft
f, t, Sxx = signal.spectrogram(X, sample_rate,window = signal.blackman(N),nfft=N)
plt.figure()
plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
#plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [seg]')
plt.title('Spectrogram with scipy.signal',size=16);

plt.show()

sd.play(X)
