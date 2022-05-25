import numpy as np

import librosa
from scipy import signal
from matplotlib import pyplot as plt
import sounddevice as sd
import os
import pywt
import pywt.data
from pydub import AudioSegment














listOfFiles1 = os.listdir("noisy_cutdown")

listOfFiles = os.listdir("noisy_murmur")

#_____________________________________

B = 1

if B==0:
    for file in listOfFiles:
        print(listOfFiles[0])
        sound = AudioSegment.from_file("noisy_murmur/"+file)

        s2_half = sound[:3000]
        print(s2_half)

        print("done split")
        # create a new file "first_half.mp3":
        
        s2_half.export("noisy_cutdown/s1s2_noise"+file, format="wav")




#______________________________________



file_name = "noisy_cutdown/"+listOfFiles1[12]
X, sample_rate = librosa.load(file_name)
file_name1 = "extra_normal_s1s2/"+'s1s2normal__109_1305653646620_C.wav'
X1, sample_rate1 = librosa.load(file_name1)
np.random.seed(0)

y = signal.wiener(X,  X.size,np.average(X1))
mf = signal.medfilt(X)

print(X)
x = np.sin(X) + .000001 * np.random.normal(size=X.size)
c = np.sin(X) + .001 * np.random.normal(size=X.size)

plt.figure(figsize=(10, 7))
plt.plot(x, label='Original signal')

plt.plot(X, label='Original')
plt.plot(signal.wiener(y), label='wiener: wiener filter')

plt.legend(loc='best')
plt.show()


sd.play(signal.wiener(y),sample_rate)