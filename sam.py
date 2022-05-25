# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 00:34:47 2019

@author: samir filfil
"""

import os
import librosa
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from pydub import AudioSegment
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

listOfFiles = os.listdir("extra normal")



B = 1

if B==0:
    for file in listOfFiles:
        print(listOfFiles[0])
        sound = AudioSegment.from_file("extra normal/"+file)
        halfway_point = len(sound) // 2
        print(len(sound) )
        print(len(sound) //2)
        s2_half = sound[:halfway_point]
        print("done split")
        # create a new file "first_half.mp3":
        
        s2_half.export("extra_normal_s1s2/s1s2"+file, format="wav")

def decodeFolder(category):
	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,193))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

def extract_feature(file_name):
	print("Extracting "+file_name+" ...")


	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	zcr = np.mean(librosa.feature.zero_crossing_rate(X).T,axis=0)
	return np.hstack((mfccs,chroma,mel,contrast,tonnetz,zcr))

#train data
normal_sounds = decodeFolder("MR_S1&S2")
normal_labels = [0 for items in normal_sounds]
murmur_sounds = decodeFolder("S1&S2")
murmur_labels = [1 for items in murmur_sounds]
train_sounds = np.concatenate((normal_sounds, murmur_sounds))
train_labels = np.concatenate((normal_labels, murmur_labels))

#test_data
test_sound = decodeFolder("test")


clf =svm.SVC()
clf.fit(train_sounds,train_labels)
print("training done")
print(clf.predict(test_sound))
X = train_sounds
y = train_labels
print (X)
print(y)

model = Sequential()
model.add(Dense(12, input_dim=193, activation='relu'))
model.add(Dense(193, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))\

predictions = model.predict_classes(y)
# summarize the first 5 cases
for i in range(6):
	print('%s => %d (expected %d)' % (X[i].tolist(),  y[i]))




