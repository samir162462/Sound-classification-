# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:09:12 2020

@author: samir filfil
"""

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
#listOfFiles = os.listdir("Trace")
print('sss')



from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout



from keras.utils.np_utils import to_categorical

def split_wav_audio(audio,sec):
    # = AudioSegment.from_file("Trace/"+file)
    s2_half = audio[:sec*1000]
    # create a new file "first_half.mp3":
    # s2_half.export("ETrace/"+file, format="wav")
    return s2_half


def set_peaks(y):
       
        
        x = y
        peaks, properties = find_peaks(x, prominence=1, width=80)
        print(peaks)
        properties["prominences"], properties["widths"]
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
                   ymax = x[peaks], color = "C1")
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                   xmax=properties["right_ips"], color = "C1")
        plt.show()
        
        
        s_arr = np.zeros(len(peaks)-1)
        pointer1 = 0 
        pointer2 = 0 
        counter = 0 
        
        s=0
        pointer1 = peaks[0]
        pointer2 = peaks[1] 
        for i in range(len(peaks)-1):
                
                if pointer2-pointer1>1000:
                        s_arr[counter] = pointer1
                        counter +=1
                        s_arr[counter] = pointer2
                        counter +=1
                        pointer1 = pointer2
                        pointer2 = peaks[i]
                        i=i+1
                        
                else:
                        
                        pointer2 = peaks[i]
                        print(s_arr)
        s_arr.astype(int)
            
        uniques = np.unique(s_arr)
        print(uniques[1])
        shift = 0
        for i in range(len(uniques)-1):
                
                if uniques[i+1]-uniques[i]<1000:
                        continue
                else:
                        print(i)
        
                        plt.plot(y[int(uniques[i]):int(uniques[i+1])])
                        plt.show()


def decodeFolder(category):

	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,60))
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
	set_peaks(X)
    
	tempo, beats = librosa.beat.beat_track(y=X, sr=sample_rate, hop_length=512)

	beat_times = librosa.frames_to_time(beats, sr=sample_rate, hop_length=512)
#	print(beats)
#	#mfccsbb = np.mean(librosa.feature.mfcc(y=beats, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
#	print(np.mean(stft[5]))
    
	tonnets = heapq.nlargest(50,tonnets)
    
    
	sr = sample_rate
	tempo, beats = librosa.beat.beat_track(y=X, sr=sr, hop_length=512)
	beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
	cqt = np.abs(librosa.cqt(X, sr=sr, hop_length=512))
	subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
	subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)
	print(subseg) 
	r1=return_audio_in_time(X,sr,subseg_t[0],subseg_t[1])
	r2=return_audio_in_time(X,sr,subseg_t[1],subseg_t[2])
	r3=return_audio_in_time(X,sr,subseg_t[2],subseg_t[3])
	mfccs1 = np.mean(librosa.feature.mfcc(y=r1, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	mfccs2 = np.mean(librosa.feature.mfcc(y=r2, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	mfccs3 = np.mean(librosa.feature.mfcc(y=r3, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	mfccs1 = heapq.nlargest(20,mfccs1)
	mfccs2 = heapq.nlargest(20,mfccs2)
	mfccs3 = heapq.nlargest(20,mfccs3)

	print(mfccs1,mfccs2,mfccs3)
#	print(subseg_t)
	#return np.hstack((mfccs,chroma,mel,contrast,tonnetz,tonnets))
    
    
    
    
    #print(return_audio_in_time(y,sr,0,2))

	return np.hstack((mfccs1,mfccs2,mfccs3))


#train data

def return_audio_in_time(audio,sr,time_start,time_end):
       
        t1 =time_start*sr
        t1 = int(t1)
        t2=time_end*sr
        t2 = int(t2)
        audio = audio[t1:t2]
        print(librosa.get_duration(y=audio,sr=sr))
        return audio
                
        
        

def get_roc(y_test,y_score,n_classes):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(1):
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
        lw = 1
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def get_accuarcy(predict_x,test_x,predict_y,test_y,train_labels,train_sounds,name):
        target_names = ['Normal ', 'Upnormal']
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
        full_prediction = np.concatenate((predict_x, predict_y))
        full_labels = np.concatenate((predx_label, predy_label))
        print(classification_report(full_labels, full_prediction, labels=[0, 1],target_names=target_names ))
        
        full_test_features = np.concatenate((test_x, test_y))
        
        #reshaping to 2D 
        train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,30,2))
        full_test_features=np.reshape(full_test_features,(full_test_features.shape[0],  1,30,2))
        print([train_labels.shape,full_test_features.shape])
        
        
        #get_roc(test_y,predict_y,3)
        cnn_model(train_sounds,train_labels,full_test_features,full_labels)



#       
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#
#        for i in range(len(full_test_features)):
#              arrays_sounds[i]=np.median(full_test_features[i])
#              if i >len(full_test_features)/2:
#                ax.scatter(i, arrays_sounds[i],color='red')
#
#              else:
#                ax.scatter(i, arrays_sounds[i],color='blue')
#
##        print(classification_report(pred_label, predict_y, labels=[0, 1],target_names=target_names ))
#        #plt.plot(arrays_sounds)
#        plt.ylabel('Mean of features')
#        plt.xlabel('Number of features')
#        plt.show()
#        


#model training and validation
import pathlib
def run_model_file(train_y,test_y):
    file = pathlib.Path("finalized_model1.sav")
    if file.exists ():
        print ("File exist")
        # load the model from disk
        loaded_model = pickle.load(open(file, 'rb'))
        test_sound = decodeFolder("N_New")    
        result = loaded_model.predict(test_sound)
        

        
    else:
        print ("File not exist")
        normal_sounds = decodeFolder("train_x")
        normal_labels = [0 for items in normal_sounds]
        murmur_sounds = decodeFolder(train_y)
        murmur_labels = [1 for items in murmur_sounds]
        train_sounds = np.concatenate((normal_sounds, murmur_sounds))
        train_labels = np.concatenate((normal_labels, murmur_labels))
        #test_data
        test_x = decodeFolder("test_x")
        
        test_y = decodeFolder(test_y)
    
    

        
        #clf1 = DecisionTreeClassifier(max_depth=4)
        clf0 = LogisticRegression(random_state=0)        
        clf1 = tree.DecisionTreeClassifier()
        
        clf2 = KNeighborsClassifier(n_neighbors=7)
        clf3 = SVC(gamma=.1, kernel='rbf', probability=True)

        eclf = VotingClassifier(estimators=[ ('DT', clf1),('lr', clf0),
                                            ('svc', clf3)],
                                voting='soft')
        #clf1.fit(train_sounds, train_labels)
        clf0.fit(train_sounds, train_labels)
        clf1.fit(train_sounds, train_labels)
        clf2.fit(train_sounds, train_labels)
        clf3.fit(train_sounds, train_labels)
        

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
#        print("Y - VotingClassifier :",eclf.predict(test_y))        
        
        get_accuarcy(clf0.predict(test_x),test_x,clf0.predict(test_y),test_y,train_labels,train_sounds,'LogisticRegression')
        get_accuarcy(clf1.predict(test_x),test_x,clf1.predict(test_y),test_y,train_labels,train_sounds,'Decision Trees')
        get_accuarcy(clf2.predict(test_x),test_x,clf2.predict(test_y),test_y,train_labels,train_sounds,'KNeighborsClassifier')
        get_accuarcy(clf3.predict(test_x),test_x,clf3.predict(test_y),test_y,train_labels,train_sounds,'SVM')
        get_accuarcy(eclf.predict(test_x),test_x,eclf.predict(test_y),test_y,train_labels,train_sounds,'VotingClassifier')

        
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


        
run_model_file('upnormal1/train_y_mr','upnormal1/test_y_mr')
#rmf_mvp = run_model_file('upnormal/train_y_mvp','upnormal/test_y_mvp')
#rmf_as = run_model_file('upnormal/train_y_as','upnormal/test_y_as')
#rmf_ms = run_model_file('upnormal/train_y_ms','upnormal/test_y_ms')
#print('acc of MR  : ',rmf_mr)
#print('acc of MVP : ',rmf_mvp)
#print('acc of AS  : ',rmf_as)
#print('acc of MS  : ',rmf_ms)


#----- model end 

#---- deep learning


def cnn_model(x_train,y_train,x_test,y_test):
        #forming model
        model=Sequential()
        #adding layers and forming the model
        model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(1,30,2)))
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

