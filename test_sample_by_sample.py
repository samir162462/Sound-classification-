# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 00:45:10 2020

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
import matplotlib.patches as mpatches



Err_samples_num = 50


def decodeFolder(category,i):

	print("Starting decoding folder "+category+" ..."+str(i))
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,1))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename,i)
		arrays_sound = np.vstack((arrays_sound,features_sound[i]))
	return arrays_sound

def extract_feature(file_name ,i):        
	#print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
#	X = split_wav_audio(X,7)
    #fft_f = fft(X)
    #fft_sf = np.sort(sam)[::-1]
    #fft_fn =sam[0:100]
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	mfccs_cqt = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5

	mfccs = heapq.nlargest(30,mfccs)	
    
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#12 feature   ->3
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)     #128 feature  ->9 out
	mel = heapq.nlargest(90,mel)     
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) # 7 features ->7
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0) #6
	#cqt = np.mean(librosa.core.cqt(mfccs_cqt, sr=sample_rate).T,axis=0)#384
	cqt = librosa.feature.poly_features(X, sr=sample_rate)#384
	ii = 0
	c = np.array(8)
	#cqt = cqt[40:70]
    

	#print((mfccs_cqt.shape))
    

	#return np.hstack((mfccs,chroma,mel,contrast,tonnetz,tonnets))
	return np.hstack(cqt_p)



        
def createList(r1, r2): 
    return [item for item in range(r1, r2+1,2)] 

for i in range(0,30):
        normal_sounds = decodeFolder("train_x",i)
        normal_test = decodeFolder("test_x",i)
        
        train_sounds_n = np.concatenate((normal_sounds, normal_test))
        
        #mr
        upnormal_mr_sound = decodeFolder("upnormal/train_y_mr",i)
        upnormal_mr_test = decodeFolder("upnormal/test_y_mr",i)
        
        train_sounds_mr_sounds = np.concatenate((upnormal_mr_sound, upnormal_mr_test))
        
        #mvp
        upnormal_mvp_sound = decodeFolder("upnormal/train_y_mvp",i)
        upnormal_mvp_test = decodeFolder("upnormal/test_y_mvp",i)
        
        train_sounds_mvp_sounds = np.concatenate((upnormal_mvp_sound, upnormal_mvp_test))
        
        #AS
        upnormal_as_sound = decodeFolder("upnormal/train_y_as",i)
        upnormal_as_test = decodeFolder("upnormal/test_y_as",i)
        
        train_sounds_as_sounds = np.concatenate((upnormal_as_sound, upnormal_as_test))
        
        
        #ms
        upnormal_ms_sound = decodeFolder("upnormal/train_y_ms",i)
        upnormal_ms_test = decodeFolder("upnormal/test_y_ms",i)
        
        train_sounds_ms_sounds = np.concatenate((upnormal_ms_sound, upnormal_ms_test))
        
        
        
        all_sounds=  np.concatenate((train_sounds_n, train_sounds_mr_sounds))
        all_sounds=  np.concatenate((all_sounds, train_sounds_mvp_sounds))
        all_sounds=  np.concatenate((all_sounds, train_sounds_as_sounds))
        all_sounds=  np.concatenate((all_sounds, train_sounds_ms_sounds))
        
        from numpy import savetxt
        
        
        savetxt('beat feature/beat'+str(i)+'.csv', all_sounds, delimiter=',')
        #savetxt('chroma features/L-Chorma'+str(i)+'.csv', all_sounds, delimiter=',')
        
        print(len(all_sounds))
        fig = plt.figure(figsize=(200,20),dpi=150)

        ax = fig.add_subplot(111)
        ax = plt.gca()
        for s in range(len(all_sounds)):
                if s >=0 and s<200:
                        ax.scatter(s, all_sounds[s],color='red')
                elif(s >=200 and s<400):
                        ax.scatter(s, all_sounds[s],color='blue')
                elif(s >=400 and s<600):
                        ax.scatter(s, all_sounds[s],color='grey')
                elif(s >=600 and s<800):
                        ax.scatter(s, all_sounds[s],color='green')
                elif(s >=800 ):
                        ax.scatter(s, all_sounds[s],color='orange')                
               #plt.plot(arrays_sounds)
        one = mpatches.Patch(facecolor='red', label='Normal', linewidth = 0.5, edgecolor = 'black')
        two = mpatches.Patch(facecolor='blue', label = 'MR', linewidth = 0.5, edgecolor = 'black')
        three = mpatches.Patch(facecolor='grey', label = 'MVP', linewidth = 0.5, edgecolor = 'black')
        four = mpatches.Patch(facecolor='green', label = 'AS', linewidth = 0.5, edgecolor = 'black')
        five = mpatches.Patch(facecolor='orange', label = 'MS', linewidth = 0.5, edgecolor = 'black')

        legend = plt.legend(handles=[one, two, three,four,five], loc = 4, fontsize = 'small', fancybox = True)
        plt.xticks(createList(0,1000))
        plt.title('L-beat feature '+str(i))
        #plt.title('L-Chorma '+str(i))
        plt.ylabel('Value')
  
        plt.xlabel('Number of sounds')
        plt.grid()
        plt.savefig('beat feature/L-beat'+str(i))        
        #plt.savefig('chroma features/L-Chorma'+str(i))        
        print("image saved well")

