# -*- coding: utf-8 -*-
"""
Created on Tue Oct 2 16:45:25 2019

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

#listOfFiles = os.listdir("Trace")
print('sss')




def split_wav_audio(audio,sec):
    # = AudioSegment.from_file("Trace/"+file)
    s2_half = audio[:sec*1000]
    # create a new file "first_half.mp3":
    # s2_half.export("ETrace/"+file, format="wav")
    return s2_half

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


#train data
    

#model training and validation
import pathlib
def run_model_file(train_y,test_y):
    file = pathlib.Path("finalized_model1.sav")
    if file.exists ():
        print ("File exist")
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        test_sound = decodeFolder("N_New")    
        result = loaded_model.predict(test_sound)
        
        from sklearn.metrics import accuracy_score
        acs = accuracy_score(result, result)
        print(result)
        print(acs)
        
    else:
        print ("File not exist")
        normal_sounds = decodeFolder("train_x")
        display(normal_sounds)
        normal_labels = [0 for items in normal_sounds]
        murmur_sounds = decodeFolder(train_y)
        murmur_labels = [1 for items in murmur_sounds]
        train_sounds = np.concatenate((normal_sounds, murmur_sounds))
        train_labels = np.concatenate((normal_labels, murmur_labels))
        #test_data
        test_x = decodeFolder("test_x")
        test_y = decodeFolder(test_y)
    
    
        clf =svm.SVC()
        clf.fit(train_sounds,train_labels)
        print("training done")
        print(clf.predict(test_x))
        predict_x = clf.predict(test_x)
        plt.plot(clf.predict(test_x))
        print(clf.predict(test_y))
        predict_y = clf.predict(test_y)
        plt.plot(clf.predict(test_y))
        countx =0
        county =0
        for i in predict_x:
            if (i==0):
                countx+=1
                
        for i in predict_y:
            if (i==1):
                county+=1
        test_x_len = len(test_x)
        test_y_len = len(test_y)
        print('lose in x_test(normal) = ',test_x_len-countx)
        print('lose in y_test(upnormal) = ',test_y_len-county)
        print('accuarcy of x file = ',(countx*100)/test_x_len)
        print('accuarcy of y file = ',(county*100)/test_y_len)
        print(' net accuarcy  = ',(((county*100)/test_y_len)+((countx*100)/test_x_len))/2)
        
    return (((county*100)/test_y_len)+((countx*100)/test_x_len))/2
    # save the model to disk
  #  filename = 'finalized_model1.sav'
   # pickle.dump(clf, open(filename, 'wb'))
 
# some time later...

rmf_mr = run_model_file('upnormal/train_y_mr','upnormal/test_y_mr')
#rmf_mvp = run_model_file('upnormal/train_y_mvp','upnormal/test_y_mvp')
#rmf_as = run_model_file('upnormal/train_y_as','upnormal/test_y_as')
#rmf_ms = run_model_file('upnormal/train_y_ms','upnormal/test_y_ms')
print('acc of MR  : ',rmf_mr)
#print('acc of MVP : ',rmf_mvp)
#print('acc of AS  : ',rmf_as)
#print('acc of MS  : ',rmf_ms)


#----- model end 

spctro_dis =1

if spctro_dis ==0:
    ts = os.listdir("test")
    fig = plt.figure(figsize=(10, 6), dpi=300)
    for i in ts:
       
        Audiodata, sample_rate = librosa.load("test/"+i)        
        N = 512 #Number of point in the fft
        f, t, Sxx = signal.spectrogram(Audiodata, sample_rate,window = signal.blackman(N),nfft=N)
       
        plt.subplot(3,3,3)
        plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
        #plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [seg]')
        plt.title(i,size=16);
        
        plt.show()
    fig.savefig("sounds.png", dpi = 300) # save figure

