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

ts = os.listdir("ETrace")

from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


for i in ts:
    x="";
    fig = plt.figure()
    Audiodata, sample_rate = librosa.load("Trace/"+i)        
    # Filter requirements.
    order = 6
    fs = sample_rate       # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz
    
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    
    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    
    
    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0         # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    
    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(Audiodata, cutoff, fs, order)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, Audiodata, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    
    plt.subplots_adjust(hspace=0.35)
    plt.show()
    
             
        





