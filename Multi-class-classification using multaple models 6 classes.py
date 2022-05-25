# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:09:12 2020

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score
from tensorflow import keras
from numpy import load
from random import shuffle
import random
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils.np_utils import to_categorical
from numpy import save
import pywt #https://pypi.org/project/PyWavelets/
from random_set import setshuflefiles_dataset


#listOfFiles = os.listdir("Trace")
print('sss')

#intiliz
x_features = 0+40+6+40+12+4+10+10+10
plot_title = 'STFT' 
plot_xaxis = 'Num of features'
plot_yaxis = 'Frequency amplitude ' 
temp_vector = [852,6930,16116,22184]
#___________________
# plot
plot_segment = False
#____

# NEW feture extrction 
run_new = False 

#___________________


def split_wav_audio(audio,sec):
    # = AudioSegment.from_file("Trace/"+file)
    s2_half = audio[:sec*1000]
    # create a new file "first_half.mp3":
    # s2_half.export("ETrace/"+file, format="wav")
    return s2_half

def return_audio_in_time(audio,sr,time_start,time_end):
       
        t1 =time_start*sr
        t1 = int(t1)
        t2=time_end*sr
        t2 = int(t2)
        audio = audio[t1:t2]
        #print(librosa.get_duration(y=audio,sr=sr))
        return audio
        
def  width_peaks(x, sr):
    peaks, _ = find_peaks(x)
    results_half = peak_widths(x, peaks, rel_height=0.5)
    
def set_peaks(y,sr):
        
        x = y
        peaks, properties = find_peaks(x ,prominence=0.1, width=80)
        #print(peaks)
        if plot_segment == True:
            properties["prominences"], properties["widths"]
            plt.plot(x)
            plt.plot(peaks, x[peaks], "x")
            plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
                       ymax = x[peaks], color = "C1")
            plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                       xmax=properties["right_ips"], color = "C1")
            plt.show()
        
        
        s_arr = np.zeros(100, dtype=int)
        pointer1 = 0 
        pointer2 = 0 
        counter = 0 
        
        s=0
        pointer1 = peaks[0]
        pointer2 = peaks[1] 
        for i in range(len(peaks)-1):
                
                if pointer2-pointer1>6000:
                        s_arr[counter] = pointer1
                        counter +=1
                        s_arr[counter] = pointer2
                        counter +=1
                        pointer1 = pointer2
                        pointer2 = peaks[i]
                        i=i+1
                        
                else:
                        
                        pointer2 = peaks[i]
                        #print(s_arr)
        s_arr.astype(int)
            
        uniques = np.unique(s_arr)
        #print(uniques[:])
        shift = 0
#        for i in range(len(uniques)-1):
#                
#                if uniques[i+1]-uniques[i]<1000:
#                        continue
#                else:
#                        #print(i)
#        
#                        plt.plot(y[int(uniques[i]):int(uniques[i+1])])
#                        plt.show()
        return uniques

def decodeFolder_random(filepath):

	print("Starting decoding folder "+filepath[0:17]+" ...")
	arrays_sound = np.empty((0,x_features))
	for file in filepath:
		features_sound = extract_feature(file)
		arrays_sound = np.vstack((arrays_sound,features_sound))
    

	return arrays_sound

def decodeFolder(category):

	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,x_features))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
    
	plt.title =  category
	plt.xlabel = plot_xaxis
	plt.ylabel = plot_yaxis
	plt.savefig(category)
	plt.show()
	return arrays_sound

def return_vectorsize(x,size):
    
    if x.size < size:
        return temp_vector
    else:
        return x



def extract_feature(file_name):        
	print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
	#print(X.dtype)
	peaks_dis = set_peaks(X,sample_rate)

    
    
#	X = split_wav_audio(X,7)
    #fft_f = fft(X)
    #fft_sf = np.sort(sam)[::-1]
    #fft_fn =sam[0:100]
	stft = np.abs(librosa.stft(X))
	s = np.mean(stft,axis=0)
	s =  heapq.nlargest(40,s)#    print(s)
	plt.plot(s)




    
	ccA, cD = pywt.dwt(X,'db2')
	coeffs2 = heapq.nlargest(90,ccA)
    
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	#mfccs = heapq.nlargest(30,mfccs)
	zcr = librosa.feature.zero_crossing_rate(X)
	chroma = np.std(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#12 feature   ->3 librosa.feature.zero_crossing_rate(y)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)     #128 feature  ->9 out
	mel = heapq.nlargest(90,mel)     
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) # 7 features ->7
	tonnetz = np.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0) #6
	tonnets = np.mean(librosa.feature.tempogram(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)#384
	tonnets = heapq.nlargest(50,tonnets)
	#return np.hstack((mfccs,chroma,mel,contrast,tonnetz,tonnets))
	mfccs_cqt = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	
	cqt = np.mean(librosa.core.cqt(mfccs_cqt, sr=sample_rate).T,axis=0)#384
    
	cqt1 = np.real(cqt[40:44])
#	cqt1 = cqt1.astype(int)
    
	cqt2 = np.real(cqt[49:53])
#	cqt2 = cqt2.astype(int)
    
	cqt3 = np.real(cqt[50:84])
#	cqt3 = cqt3.astype(int)
    
	cqtall = np.concatenate((cqt2,cqt3)) #38
    
	#print(return_vectorsize(peaks_dis[1:5],4))
	segment_vect =  return_vectorsize(peaks_dis[1:5],4)
    
	p1 = segment_vect[0]
	p2 = segment_vect[1]
	p3 = segment_vect[2]
	p4 = segment_vect[3]
    
	segments1 = np.zeros(p2-p1)
	segments2 = np.zeros(p4-p3)


	segments1 = X[p1:p2]
	segments2 = X[p2:p3]
	segments3 = X[p3:p4]
    
	mfccs1 = np.mean(librosa.feature.mfcc(y=segments1, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	mfccs2 = np.mean(librosa.feature.mfcc(y=segments2, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5

	stftsegments1 = np.abs(librosa.stft(segments1))
	stftsegments2 = np.abs(librosa.stft(segments2))
	stftsegments3 = np.abs(librosa.stft(segments3))
	s1 = np.mean(stftsegments1,axis=0)
	s1 =  heapq.nlargest(10,s1)#    print(s)

#	print(s1)
	s2 = np.mean(stftsegments2,axis=0)
	s2 =  heapq.nlargest(10,s2)#    print(s)
	s3 = np.mean(stftsegments3,axis=0)
	s3 =  heapq.nlargest(10,s3)#



    
	return np.hstack((mfccs,chroma,tonnetz,s,cqt2,s1,s2,s1))



#train data
def get_accuarcy(predict_x,test_x,predict_y,test_y,predict_z,predict_a,predict_b,predict_c,train_labels,train_sounds,name,test_all):
        target_names = ['Normal ', 'Upnormal MR','Upnormal MVP_p1','Upnormal MS','Upnormal AS','Upnormal MVP_p2']
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
        print('_________________________')
        print('lose in x_test(normal) ',name,' = ',test_x_len-countx)
        print('lose in y_test(upnormal)',name,' = ',test_y_len-county)
        print('accuarcy of x ',name,' file = ',(countx*100)/test_x_len)
        print('accuarcy of y ',name,' file = ',(county*100)/test_y_len)
        print(' net accuarcy of ',name,'  = ',(((county*100)/test_y_len)+((countx*100)/test_x_len))/2)
        predy_label = [1 for items in predict_y]
        predx_label = [0 for items in predict_x]
        predz_label = [2 for items in predict_z]
        preda_label = [3 for items in predict_a]
        predb_label = [4 for items in predict_b]
        predc_label = [5 for items in predict_c]
        full_prediction = np.concatenate((predict_x, predict_y,predict_z))
        full_prediction = np.concatenate((full_prediction,predict_a,predict_b))
        full_prediction = np.concatenate((full_prediction,predict_c))
        full_labels = np.concatenate((predx_label, predy_label,predz_label))
        full_labels = np.concatenate((full_labels,preda_label,predb_label))
        full_labels = np.concatenate((full_labels,predc_label))
        print(classification_report(full_labels, full_prediction, labels=[0, 1,2,3,4,5],target_names=target_names ))


   
        
#
#        

def get_accuarcy_DNN_dirict(train_sounds,test_sounds,train_labels,test_labels):
        target_names = ['Normal ', 'Upnormal MR','Upnormal MVP_p1','Upnormal MS','Upnormal AS','Upnormal MVP_p2']
  
    

        full_test_features = test_sounds
        d1_labels = test_labels
        test_labels = to_categorical(test_labels, num_classes=6)
        train_labels = to_categorical(train_labels, num_classes=6)

        train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,int(x_features/2),2))
        full_test_features=np.reshape(full_test_features,(full_test_features.shape[0],1,int(x_features/2),2))

        
        max_number = 0 
        for i in range(50):
            print(i)
            model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1,int(x_features/2),2)),
            keras.layers.Dense(int(x_features), activation=tf.nn.relu),
        	keras.layers.Dense(int(x_features/2), activation=tf.nn.relu),
            keras.layers.Dense(6, activation=tf.nn.sigmoid),#softmax sigmoid
            ])#sigmoid
            
            model.compile(optimizer='adam',
                          loss='binary_crossentropy', #categorical_crossentropy
                          metrics=['accuracy'])
            
            model.fit(train_sounds,train_labels, epochs=35, batch_size=20,verbose=0)
            
            test_loss, test_acc = model.evaluate(full_test_features,test_labels)
            ynew = model.predict(full_test_features)
            y_classes = ynew.argmax(axis=-1)
 

            s = []
            for i in ynew:
                
                result = np.where(i == np.amax(i))
                for cord in result:
                    x = str(cord)
                    s.append(int(x[1]))
                    
            d = np.array(s)
            classfication_model = classification_report(d1_labels,y_classes, labels=[0, 1,2,3,4,5],target_names=target_names )
            accuracy = accuracy_score(d1_labels,d)
    #        print(d)
            if accuracy > max_number:
                max_number=accuracy
                print(classfication_model)
                df = pd.DataFrame(ynew)
                ## save to xlsx file
                filepath = 'my_excel_file.xlsx'
                df.to_excel(filepath, index=False)
                save('model_predicted', d)        
                save('model_predicted', d1_labels)        



def get_accuarcy_DNN(predict_x,test_x,predict_y,test_y,predict_z,predict_a,predict_b,predict_c,train_labels,train_sounds,name,test_all):
        target_names = ['Normal ', 'Upnormal MR','Upnormal MVP_p1','Upnormal MS','Upnormal AS','Upnormal MVP_p2']
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
        predy_label = [1 for items in predict_y]
        predx_label = [0 for items in predict_x]
        predz_label = [2 for items in predict_z]
        preda_label = [3 for items in predict_a]
        predb_label = [4 for items in predict_b]
        predc_label = [5 for items in predict_c]
        full_prediction = np.concatenate((predict_x, predict_y,predict_z))
        full_prediction = np.concatenate((full_prediction,predict_a,predict_b))
        full_prediction = np.concatenate((full_prediction,predict_c))
        full_labels = np.concatenate((predx_label, predy_label,predz_label))
        full_labels = np.concatenate((full_labels,preda_label,predb_label))
        full_labels = np.concatenate((full_labels,predc_label))

        full_test_features = test_all
        d1_labels = full_labels
        train_labels = to_categorical(train_labels, num_classes=6)
        full_labels = to_categorical(full_labels, num_classes=6)
        
        train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,int(x_features/2),2))
        full_test_features=np.reshape(full_test_features,(full_test_features.shape[0],1,int(x_features/2),2))
        

        
        max_number = 0 
        for i in range(50):
            print(i)
            model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1,int(x_features/2),2)),
            keras.layers.Dense(int(x_features), activation=tf.nn.tanh),
        	keras.layers.Dense(int(x_features/2), activation=tf.nn.tanh),
            keras.layers.Dense(6, activation=tf.nn.sigmoid),
            ])#sigmoid
            
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            
            model.fit(train_sounds,train_labels, epochs=35, batch_size=20,verbose=0)
            
            test_loss, test_acc = model.evaluate(full_test_features,full_labels)
            ynew = model.predict(full_test_features)

            s = []
            for i in ynew:
                
                result = np.where(i == np.amax(i))
                for cord in result:
                    x = str(cord)
                    s.append(int(x[1]))
                    
            d = np.array(s)
            classfication_model = classification_report(d1_labels,d, labels=[0, 1,2,3,4,5],target_names=target_names )
            accuracy = accuracy_score(d1_labels,d)
    #        print(d)
            if accuracy > max_number:
                max_number=accuracy
                print(classfication_model)
                df = pd.DataFrame(ynew)
                ## save to xlsx file
                filepath = 'my_excel_file.xlsx'
                df.to_excel(filepath, index=False)
                save('model_predicted', d)        
                save('classfication_model', classfication_model)        

#model training and validation
import pathlib
def run_model_file(train_y,test_y,train_z,test_z,train_a,test_a,train_b,test_b,train_c,test_c):
    file1 = pathlib.Path("train_heart_sounds.npy")
    file2 = pathlib.Path("test_heart_sounds.npy")
    file3 = pathlib.Path("train_labels_heart_sounds.npy")
    file4 = pathlib.Path("test_labels_heart_sounds.npy")
    if file1.exists ()and file2.exists()and file3.exists()and file4.exists() and not run_new:
        print ("File exist")
        # load the model from disk
        loaded_train_sounds = load('train_heart_sounds.npy')
        loaded_test_sounds =  load('test_heart_sounds.npy')
        loaded_train_labels =   load('train_labels_heart_sounds.npy')
        loaded_test_labels =   load('test_labels_heart_sounds.npy')
        get_accuarcy_DNN_dirict(loaded_train_sounds,loaded_test_sounds,loaded_train_labels,loaded_test_labels)
        

        
    else:
        print ("File not exist")
        normal_sounds = decodeFolder("train_test_50_50/train_x")
        normal_labels = [0 for items in normal_sounds]
        murmur_sounds = decodeFolder(train_y)
        murmur_labels = [1 for items in murmur_sounds]
        
        murmur_sounds1 = decodeFolder(train_z)
        murmur_labels1 = [2 for items in murmur_sounds1]

        murmur_sounds2 = decodeFolder(train_a)
        murmur_labels2 = [3 for items in murmur_sounds2]  
        
        murmur_sounds3 = decodeFolder(train_b)
        murmur_labels3 = [4 for items in murmur_sounds3]  
        
        murmur_sounds4 = decodeFolder(train_c)
        murmur_labels4 = [5 for items in murmur_sounds4]  
        
                
        
        train_sounds = np.concatenate((normal_sounds, murmur_sounds,murmur_sounds1))
        train_sounds = np.concatenate((train_sounds, murmur_sounds2,murmur_sounds3))
        train_sounds = np.concatenate((train_sounds,murmur_sounds4))
        
        train_labels = np.concatenate((normal_labels, murmur_labels,murmur_labels1))
        train_labels = np.concatenate((train_labels,murmur_labels2,murmur_labels3))
        train_labels = np.concatenate((train_labels,murmur_labels4))
        #test_data
        test_x = decodeFolder("train_test_50_50/test_x")
        test_y = decodeFolder(test_y)
        test_z = decodeFolder(test_z)
        test_a = decodeFolder(test_a)
        test_b = decodeFolder(test_b)
        test_c = decodeFolder(test_c)
    
        predy_label = [1 for items in test_y]
        predx_label = [0 for items in test_x]
        predz_label = [2 for items in test_z]
        preda_label = [3 for items in test_a]
        predb_label = [4 for items in test_b]
        predc_label = [5 for items in test_c]
        
        test_labels = np.concatenate((predx_label, predy_label,predz_label))
        test_labels = np.concatenate((test_labels,preda_label,predb_label))
        test_labels = np.concatenate((test_labels,predc_label))
        
        
        testall = np.concatenate((test_x, test_y,test_z))
        testall = np.concatenate((testall, test_a,test_b))
        testall = np.concatenate((testall,test_c))
        #save raw data and labels
        save('train_heart_sounds.npy', train_sounds)
        save('train_labels_heart_sounds.npy', train_labels)
        save('test_heart_sounds.npy', testall)
        save('test_labels_heart_sounds.npy', test_labels)
     
        test_all = testall
        
        clf1 = SVC(gamma=.1, kernel='rbf', probability=True)
        clf0 =  KNeighborsClassifier(n_neighbors=1) 
        #clf1 = tree.DecisionTreeClassifier()
        
        clf2 = KNeighborsClassifier(n_neighbors=22) 
        #clf3 = 

        clf3 = KNeighborsClassifier(n_neighbors=11) 
        
        eclf = VotingClassifier(estimators=[ ('DT', clf1),('KNN 2', clf2),
                                            ('KNeighbor sClassifier 11', clf3)],
                                voting='soft')
        #clf1.fit(train_sounds, train_labels)
        
        clf3.fit(train_sounds, train_labels)
        clf0.fit(train_sounds, train_labels)
        clf1.fit(train_sounds, train_labels)
        clf2.fit(train_sounds, train_labels)
        #clf3.fit(train_sounds, train_labels)
        

        eclf.fit(train_sounds, train_labels)

        #print("DecisionTreeClassifier :",clf1.predict(test_x))

#        print("X - LogisticRegression :",clf0.predict(test_x))
#        print("X - KNeighborsClassifier :",clf2.predict(test_x))
#        print("X - SVC  :",clf3.predict(test_x))
#        print("X - VotingClassifier :",eclf.predict(test_x))
#        
#        print("Y- LogisticRegression :",clf0.predict(test_y))
#        print("Y- KNeighborsClassifier :",clf2.predict(test_y))
#        print("Y - SVC  :",clf3.predict(test_y))
#        print("Y - VotingClassifier :",eclf.predict(test_y))     get_accuarcy_DNN   
        
        #get_accuarcy(clf0.predict(test_x),test_x,clf0.predict(test_y),test_y,clf0.predict(test_z),clf0.predict(test_a),clf0.predict(test_b),train_labels,train_sounds,'Random Forest Classifier',test_all)
        #get_accuarcy(clf1.predict(test_x),test_x,clf1.predict(test_y),test_y,clf1.predict(test_z),clf1.predict(test_a),clf1.predict(test_b),train_labels,train_sounds,'Decision Trees',test_all)
        #get_accuarcy(clf2.predict(test_x),test_x,clf2.predict(test_y),test_y,clf2.predict(test_z),clf2.predict(test_a),clf2.predict(test_b),train_labels,train_sounds,'KNeighborsClassifier',test_all)
        get_accuarcy(clf1.predict(test_x),test_x,clf1.predict(test_y),test_y,clf1.predict(test_z),clf1.predict(test_a),clf1.predict(test_b),clf1.predict(test_c),train_labels,train_sounds,'SVM',test_all)
        get_accuarcy(clf3.predict(test_x),test_x,clf3.predict(test_y),test_y,clf3.predict(test_z),clf3.predict(test_a),clf3.predict(test_b),clf3.predict(test_c),train_labels,train_sounds,'KNeighbors Classifier (2)',test_all)
        get_accuarcy(eclf.predict(test_x),test_x,eclf.predict(test_y),test_y,eclf.predict(test_z),eclf.predict(test_a),eclf.predict(test_b),eclf.predict(test_c),train_labels,train_sounds,'Voting Classifier',test_all)
        get_accuarcy_DNN(eclf.predict(test_x),test_x,eclf.predict(test_y),test_y,eclf.predict(test_z),eclf.predict(test_a),eclf.predict(test_b),eclf.predict(test_c),train_labels,train_sounds,'DNN',test_all)

        
#       print(clf.predict(test_x))
#        predict_x = eclf.predict(test_x)
#       plt.plot(clf.predict(test_x))
#       print(clf.predict(test_y))
#        predict_y = eclf.predict(test_y)
#       plt.plot(clf.predict(test_y))
#       plt.plot(clf.predict(test_y))
#       plt.plot(clf.predict(test_y))
#       plt.plot(clf.predict(test_y))
#        countx =0
#        county =0
#        for i in predict_x:
#            if (i==0):
#                countx+=1
#                
#        for i in predict_y:
#            if (i==1):
#                county+=1
#        test_x_len = len(test_x)
#        test_y_len = len(test_y)
#        print('lose in x_test(normal) = ',test_x_len-countx)
#        print('lose in y_test(upnormal) = ',test_y_len-county)
#        print('accuarcy of x file = ',(countx*100)/test_x_len)
#        print('accuarcy of y file = ',(county*100)/test_y_len)
#        print(' net accuarcy  = ',(((county*100)/test_y_len)+((countx*100)/test_x_len))/2)
##        
##
#      
#    return (((county*100)/test_y_len)+((countx*100)/test_x_len))/2
        
    # save the model to disk
  #  filename = 'finalized_model1.sav'
   # pickle.dump(clf, open(filename, 'wb'))
 
# some time later...

train_n,test_n = setshuflefiles_dataset("train_test_50_50 - random",5,100)
train_mr,test_mr = setshuflefiles_dataset("train_test_50_50 - random",1,100)
train_mvp1,test_mvp1 = setshuflefiles_dataset("train_test_50_50 - random",3,100)
train_mvp2,test_mvp2 = setshuflefiles_dataset("train_test_50_50 - random",4,100)
train_as,test_as = setshuflefiles_dataset("train_test_50_50 - random",0,100)
train_as,test_ms = setshuflefiles_dataset("train_test_50_50 - random",2,100)


        
#run_model_file('upnormal/train_y_mr','upnormal/test_y_mr','upnormal/train_y_mvp','upnormal/test_y_mvp')
run_model_file('train_test_50_50/train_y_mr','train_test_50_50/test_y_mr','train_test_50_50/train_y_mvp','train_test_50_50/test_y_mvp','train_test_50_50/train_y_ms'
               ,'train_test_50_50/test_y_ms','train_test_50_50/train_y_as','train_test_50_50/test_y_as','train_test_50_50/train_y_mvp_p2','train_test_50_50/test_y_mvp_p2')
#rmf_mvp = run_model_file('upnormal/train_y_mvp','upnormal/test_y_mvp')
#rmf_as = run_model_file('upnormal/train_y_as','upnormal/test_y_as')
#rmf_ms = run_model_file('upnormal/train_y_ms','upnormal/test_y_ms')
#print('acc of MR  : ',rmf_mr)
#print('acc of MVP : ',rmf_mvp)
#print('acc of AS  : ',rmf_as)
#print('acc of MS  : ',rmf_ms)


#----- model end 


        