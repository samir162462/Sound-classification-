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


from tensorflow import keras

from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils.np_utils import to_categorical

import pywt #https://pypi.org/project/PyWavelets/
#listOfFiles = os.listdir("Trace")
print('sss')


x_features = 0+40+6+40+12+4

plot_title = 'MFCCs '




def split_wav_audio(audio,sec):
    # = AudioSegment.from_file("Trace/"+file)
    s2_half = audio[:sec*1000]
    # create a new file "first_half.mp3":
    # s2_half.export("ETrace/"+file, format="wav")
    return s2_half

def decodeFolder(category):

	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,x_features))
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
	s = np.mean(stft,axis=0)
	print(s.shape)
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
	print(chroma)
#	cqt1 = cqt1.astype(int)
    
	cqt2 = np.real(cqt[49:53])
#	cqt2 = cqt2.astype(int)
    
	cqt3 = np.real(cqt[50:84])
#	cqt3 = cqt3.astype(int)
    
	cqtall = np.concatenate((cqt2,cqt3)) #38
    
	#print(cqtall.shape)
    
    
	return np.hstack((mfccs,chroma,tonnetz,s,cqt2))



#train data
def get_accuarcy(predict_x,test_x,predict_y,test_y,predict_z,predict_a,predict_b,train_labels,train_sounds,name,test_all):
        target_names = ['Normal ', 'Upnormal MR','Upnormal MVP','Upnormal MS','Upnormal AS']
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
        full_prediction = np.concatenate((predict_x, predict_y,predict_z))
        full_prediction = np.concatenate((full_prediction,predict_a,predict_b))
        full_labels = np.concatenate((predx_label, predy_label,predz_label))
        full_labels = np.concatenate((full_labels,preda_label,predb_label))
        print(classification_report(full_labels, full_prediction, labels=[0, 1,2,3,4],target_names=target_names ))

        full_test_features = test_all
        print([train_labels.shape,full_labels.shape])
        d1_labels = full_labels
        train_labels = to_categorical(train_labels, num_classes=5)
        full_labels = to_categorical(full_labels, num_classes=5)
        print([train_labels.shape,full_labels.shape])
        
        train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,int(x_features/2),2))
        full_test_features=np.reshape(full_test_features,(full_test_features.shape[0],1,int(x_features/2),2))
        print([train_labels.shape,full_test_features.shape])
        
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1,int(x_features/2),2)),
            keras.layers.Dense(int(x_features), activation=tf.nn.relu),
        	keras.layers.Dense(int(x_features/2), activation=tf.nn.relu),
            keras.layers.Dense(5, activation=tf.nn.sigmoid),
        ])
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(train_sounds,train_labels, epochs=50, batch_size=20)
        
        test_loss, test_acc = model.evaluate(full_test_features,full_labels)
        ynew = model.predict(full_test_features)
        print('Test accuracy:', test_acc)
        s = []
        for i in ynew:
            
            result = np.where(i == np.amax(i))
            for cord in result:
                x = str(cord)
                s.append(int(x[1]))
                
        d = np.array(s)
        print(ynew)
        print(ynew.shape)
        print(d)
        print(d.shape)
        print(full_labels)
        
        print(classification_report(d1_labels,d, labels=[0, 1,2,3,4,5],target_names=target_names ))

        # Index of top values
#        with tf.Session() as sess:
#                 indexes = tf.argmax(ynew, axis=1).eval()
#                 print(sess.run(indexes))
#        
        # prints [0 2 2]
        ## convert your array into a dataframe
        df = pd.DataFrame(np.concatenate(ynew,s))
        
        ## save to xlsx file
        
        filepath = 'my_excel_file.xlsx'
        
        df.to_excel(filepath, index=False)
                        
        
#        #get_roc(test_y,predict_y,3)
#        cnn_model(train_sounds,train_labels,full_test_features,full_labels)





#        full_test_features = np.concatenate((test_x, test_y))
#        arrays_sounds = np.empty((len(full_test_features)))
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
def run_model_file(train_y,test_y,train_z,test_z,train_a,test_a,train_b,test_b):
    file = pathlib.Path("finalized_model1.sav")
    if file.exists ():
        print ("File exist")
        # load the model from disk
        loaded_model = pickle.load(open(file, 'rb'))
        test_sound = decodeFolder("N_New")    
        result = loaded_model.predict(test_sound)
        

        
    else:
        print ("File not exist")
        normal_sounds = decodeFolder("train_test_50_50 _5 classes/train_x")
        normal_labels = [0 for items in normal_sounds]
        murmur_sounds = decodeFolder(train_y)
        murmur_labels = [1 for items in murmur_sounds]
        
        murmur_sounds1 = decodeFolder(train_z)
        murmur_labels1 = [2 for items in murmur_sounds1]

        murmur_sounds2 = decodeFolder(train_a)
        murmur_labels2 = [3 for items in murmur_sounds2]  
        
        murmur_sounds3 = decodeFolder(train_b)
        murmur_labels3 = [4 for items in murmur_sounds3]  
        

        
                
        
        train_sounds = np.concatenate((normal_sounds, murmur_sounds,murmur_sounds1))
        train_sounds = np.concatenate((train_sounds, murmur_sounds2,murmur_sounds3))
        
        train_labels = np.concatenate((normal_labels, murmur_labels,murmur_labels1))
        train_labels = np.concatenate((train_labels,murmur_labels2,murmur_labels3))
        #test_data
        test_x = decodeFolder("train_test_50_50 _5 classes/test_x")
        test_y = decodeFolder(test_y)
        test_z = decodeFolder(test_z)
        test_a = decodeFolder(test_a)
        test_b = decodeFolder(test_b)
    
        
        
        testall = np.concatenate((test_x, test_y,test_z))
        testall = np.concatenate((testall, test_a,test_b))
        
     
        test_all = testall
        
        #clf1 = DecisionTreeClassifier(max_depth=4)
        clf0 =  KNeighborsClassifier(n_neighbors=21, p = 5, metric = "euclidean") 
        #clf1 = tree.DecisionTreeClassifier()
        clf1 =  KNeighborsClassifier(n_neighbors=41 , p = 5, metric = "euclidean") 
        
        clf2 = KNeighborsClassifier(n_neighbors=2) 
        #clf3 = SVC(gamma=.1, kernel='rbf', probability=True)

        clf3 = KNeighborsClassifier(n_neighbors=2, p = 5, metric = "euclidean") 
        eclf = VotingClassifier(estimators=[ ('knn 41', clf1),('KNN 2', clf2),
                                            ('KNeighbor sClassifier 21', clf0)],
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
#        print("Y - VotingClassifier :",eclf.predict(test_y))        
        
        #get_accuarcy(clf0.predict(test_x),test_x,clf0.predict(test_y),test_y,clf0.predict(test_z),clf0.predict(test_a),clf0.predict(test_b),train_labels,train_sounds,'Random Forest Classifier',test_all)
        #get_accuarcy(clf1.predict(test_x),test_x,clf1.predict(test_y),test_y,clf1.predict(test_z),clf1.predict(test_a),clf1.predict(test_b),train_labels,train_sounds,'Decision Trees',test_all)
        #get_accuarcy(clf2.predict(test_x),test_x,clf2.predict(test_y),test_y,clf2.predict(test_z),clf2.predict(test_a),clf2.predict(test_b),train_labels,train_sounds,'KNeighborsClassifier',test_all)
        get_accuarcy(clf3.predict(test_x),test_x,clf3.predict(test_y),test_y,clf3.predict(test_z),clf3.predict(test_a),clf3.predict(test_b),train_labels,train_sounds,'KNeighbors Classifier (2)',test_all)
        get_accuarcy(eclf.predict(test_x),test_x,eclf.predict(test_y),test_y,eclf.predict(test_z),eclf.predict(test_a),eclf.predict(test_b),train_labels,train_sounds,'Voting Classifier',test_all)

        
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


        
#run_model_file('upnormal/train_y_mr','upnormal/test_y_mr','upnormal/train_y_mvp','upnormal/test_y_mvp')
run_model_file('train_test_50_50 _5 classes/train_y_mr','train_test_50_50 _5 classes/test_y_mr','train_test_50_50 _5 classes/train_y_mvp','train_test_50_50 _5 classes/test_y_mvp','train_test_50_50/train_y_ms'
               ,'train_test_50_50 _5 classes/test_y_ms','train_test_50_50 _5 classes/train_y_as','train_test_50_50 _5 classes/test_y_as')
#rmf_mvp = run_model_file('upnormal/train_y_mvp','upnormal/test_y_mvp')
#rmf_as = run_model_file('upnormal/train_y_as','upnormal/test_y_as')
#rmf_ms = run_model_file('upnormal/train_y_ms','upnormal/test_y_ms')
#print('acc of MR  : ',rmf_mr)
#print('acc of MVP : ',rmf_mvp)
#print('acc of AS  : ',rmf_as)
#print('acc of MS  : ',rmf_ms)


#----- model end 

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
        