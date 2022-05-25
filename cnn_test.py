# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:25:32 2020

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
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils.np_utils import to_categorical



def decodeFolder(category):

	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,30+12+6))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

def extract_feature(file_name):        
	print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
#	X = split_wav_audio(X,7)
    #fft_f = fft(X)
    #fft_sf = np.sort(sam)[::-1]
    #fft_fn =sam[0:100]
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	mfccs = heapq.nlargest(30,mfccs)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#12 feature   ->3
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)     #128 feature  ->9 out
	mel = heapq.nlargest(90,mel)     
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) # 7 features ->7
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0) #6
	tonnets = np.mean(librosa.feature.tempogram(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)#384
	tonnets = heapq.nlargest(50,tonnets)
	#return np.hstack((mfccs,chroma,mel,contrast,tonnetz,tonnets))
	return np.hstack((mfccs,chroma,tonnetz))

normal_sounds = decodeFolder("test/testxnm")
normal_labels = [0 for items in normal_sounds]
murmur_sounds = decodeFolder("test/testxms")
murmur_labels = [1 for items in murmur_sounds]
train_sounds = np.concatenate((normal_sounds, murmur_sounds))
train_labels = np.concatenate((normal_labels, murmur_labels))
#test_data
test_x = decodeFolder("test/testyms")
ms_labels_t = [1 for items in test_x]

test_y = decodeFolder('test/testynm')
nm_labels_t = [0 for items in test_y]

test_sounds = np.concatenate((test_x, test_y))
test_labels = np.concatenate((nm_labels_t, ms_labels_t))

print([train_sounds.shape,train_labels.shape,test_sounds.shape,test_labels.shape])
print(np.array([train_labels,np.array(20)]).shape)

#converting to one hot
#from keras.utils.np_utils import to_categorical
#train_labels = to_categorical(train_labels, num_classes=2)
#test_labels = to_categorical(test_labels, num_classes=2)
#print([train_labels.shape,test_labels.shape])

#reshaping to 2D 
train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,24,2))
test_sounds=np.reshape(test_sounds,(test_sounds.shape[0],  1,24,2))
print([train_labels.shape,test_labels.shape])

def cnn_model(x_train,y_train,x_test,y_test):
        #forming model
        model=Sequential()
        #adding layers and forming the model
        model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(1,24,2)))
        model.add(MaxPooling2D(padding="same"))
        
        model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
        model.add(MaxPooling2D(padding="same"))
        model.add(Dropout(0.3))
        
        model.add(Flatten())
        
        model.add(Dense(256,activation="relu"))
        model.add(Dropout(0.3))
        
        model.add(Dense(512,activation="relu"))
        model.add(Dropout(0.3))
        
        model.add(Dense(2,activation="softmax"))
        
        #compiling
        model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
        
        #training the model
        model.fit(x_train,y_train,batch_size=50,epochs=30,validation_data=(x_test,y_test))
        
        #train and test loss and scores respectively
        train_loss_score=model.evaluate(x_train,y_train)
        test_loss_score=model.evaluate(x_test,y_test)
        print(train_loss_score)
        print(test_loss_score)
        

#cnn_model(train_sounds,train_labels,test_sounds,test_labels)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,24,2)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_sounds, train_labels, epochs=2, batch_size=1)

test_loss, test_acc = model.evaluate(test_sounds, test_labels)
print('Test accuracy:', test_acc)
        