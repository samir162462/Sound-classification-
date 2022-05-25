# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:07:53 2019

@author: samir filfil
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
import librosa 
import sounddevice as sd



listOfFiles1 = os.listdir("noisy_cutdown")

listOfFiles = os.listdir("noisy_murmur")
file_name = "noisy_cutdown/"+listOfFiles1[12]
X, sample_rate = librosa.load(file_name)
file_name1 = "extra_normal_s1s2/"+'s1s2normal__109_1305653646620_C.wav'
X1, sample_rate1 = librosa.load(file_name1)

N = 8129  # number of samples
M = 1000  # length of Wiener filter
Om0 = 0.1*np.pi  # frequency of original signal
N0 = .1  # PSD of additive white noise

# generate original signal
s = X
# generate observed signal
g = X1
n = np.random.normal(size=N, scale=np.sqrt(N0))
x = np.convolve(s, g, mode='same') 
# estimate PSD
f, Pss = sig.csd(s, s, nperseg=M)
f, Pnn = sig.csd(n, n, nperseg=M)
# compute Wiener filter
G = np.fft.rfft(g, M)
H = 1/G * (np.abs(G)**2 / (np.abs(G)**2 + N0/Pss))
H = H * np.exp(-1j*2*np.pi/len(H)*np.arange(len(H))*(len(H)//2-8))  # shift for causal filter
h = np.fft.irfft(H)
# apply Wiener filter to observation
y = np.convolve(x, h, mode='same')

# plot (cross) PSDs
Om = np.linspace(0, np.pi, num=len(H))

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(Om, 20*np.log10(np.abs(.5*Pss)), label=r'$| \Phi_{ss}(e^{j \Omega}) |$ in dB')
plt.plot(Om, 20*np.log10(np.abs(.5*Pnn)), label=r'$| \Phi_{nn}(e^{j \Omega}) |$ in dB')
plt.title('PSDs')
plt.xlabel(r'$\Omega$')
plt.legend()
plt.axis([0, np.pi, -60, 40])
plt.grid()

# plot transfer function of Wiener filter
plt.subplot(122)
plt.plot(Om, 20*np.log10(np.abs(H)))
plt.title('Transfer function of Wiener filter')
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$| H(e^{j \Omega}) |$ in dB')
plt.axis([0, np.pi, -150, 10])
plt.grid()
plt.tight_layout()

# plot signals
idx = np.arange(500, 600)
plt.figure(figsize=(10, 4))
plt.plot(idx, x[idx], label=r'observed signal $x[k]$')
plt.plot(idx, s[idx], label=r'original signal $s[k]$')
plt.plot(idx, y[idx], label=r'estimated signal $y[k]$')
plt.title('Signals')
plt.xlabel(r'$k$')
plt.axis([idx[0], idx[-1], -1.5, 1.5])
plt.legend()
plt.grid()

sd.play(x)
