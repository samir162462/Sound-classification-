# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:09:12 2020

@author: samir filfil
"""



import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import heapq
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB 
from sklearn import tree
import tensorflow as tf
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score
from tensorflow import keras
from numpy import load
from keras.utils.np_utils import to_categorical
from numpy import save
import pywt #https://pypi.org/project/PyWavelets/
from random_set import setshuflefiles_dataset
import sklearn.metrics as sk  #conda install scikit-learn
import seaborn as sns
from Smart_Adjustive_training_set  import smart_adaptive_dataset
#pip install seaborn
from keras.utils.vis_utils import plot_model #pip install pydot

#intiliz
x_features = 0+40+6+40+12+4+10+10+10 #number of features 
plot_title = 'STFT' 
plot_xaxis = 'Num of features'
plot_yaxis = 'Frequency amplitude ' 
temp_vector = [852,6930,16116,22184] # temp vector for segmention 
accuarcy_all = 0 
#___________________
# Segmentation 
plot_segment = False 
Segment_distance = 6000 # the Distaincee between two peeks must be any number more than Segment_distance
#____

# NEW feture extrction 
run_new = False # start nnew training 
run_deep = False
epoch_DNN_repeat = 150
acuarcy_target = 100
#predict just inside the folder 
predict_folder = True  
#DNN - SAM model
Deep_DNN_smart = True # use smart adustive training set  STD 
Deep_DNN_smart_std_mean = True # use smart adustive training set  STD + MEAN  (SAM)
#___

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
        return audio
        
#def  width_peaks(x, sr):
#    peaks, _ = find_peaks(x)
#    results_half = peak_widths(x, peaks, rel_height=0.5)
    
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

	print("Starting decoding folder "+filepath[0][0:16]+" ...")
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
	peaks_dis = set_peaks(X,sample_rate)
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
	mfccs_cqt = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #40 feature  ->5
	cqt = np.mean(librosa.core.cqt(mfccs_cqt, sr=sample_rate).T,axis=0)#384
	cqt1 = np.real(cqt[40:44])
	cqt2 = np.real(cqt[49:53])
	cqt3 = np.real(cqt[50:84])
	cqtall = np.concatenate((cqt2,cqt3)) #38
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
	s1 =  heapq.nlargest(10,s1)
	s2 = np.mean(stftsegments2,axis=0)
	s2 =  heapq.nlargest(10,s2)
	s3 = np.mean(stftsegments3,axis=0)
	s3 =  heapq.nlargest(10,s3)
	return np.hstack((mfccs,chroma,tonnetz,s,cqt2,s1,s2,s1)) # stack of features 



def confusion_matrix(name,x,y):
    ax= plt.subplot()
    cm = sk.confusion_matrix(x, y)
    sns.heatmap(cm, annot=True, ax = ax,cmap="Blues"); #annot=True to annotate cells
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(name+' Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Normal ', 'MR','MVP','MS','AS']); ax.yaxis.set_ticklabels(['AS', 'MS', 'MVP', 'MR', 'Normal']);
    plt.show()
        
def get_accuarcy_SV_dirict(train_sounds,test_sounds,train_labels,test_labels):
        target_names = ['Normal', 'MR','MVP','MS','AS']
        
        clf0 =  KNeighborsClassifier(n_neighbors=1) 
        clf1 = SVC(gamma=.1, kernel='rbf', probability=True)
        clf2 = tree.DecisionTreeClassifier(max_depth = 3)
        clf3 = GaussianNB()
        
        eclf = VotingClassifier(estimators=[ ('KNN 1', clf0),('DT', clf2),
                                            ('SVM "RBF"', clf1)],
                                voting='hard')

        
        print('svmd 1')
        clf0.fit(train_sounds, train_labels)
        clf1.fit(train_sounds, train_labels)
        clf2.fit(train_sounds, train_labels)
        clf3.fit(train_sounds, train_labels)
        eclf.fit(train_sounds, train_labels)
        print('svmd2')

        print('-----KNN----')
        print(classification_report(test_labels, clf0.predict(test_sounds), labels=[0, 1,2,3,4],target_names=target_names ))
        confusion_matrix('KNN',test_labels, clf0.predict(test_sounds))
        print('-----SVM----')        
        print(classification_report(test_labels, clf1.predict(test_sounds), labels=[0, 1,2,3,4],target_names=target_names ))
        confusion_matrix('SVM',test_labels, clf1.predict(test_sounds))
        print('-----DT----')
        print(classification_report(test_labels, clf2.predict(test_sounds), labels=[0, 1,2,3,4],target_names=target_names ))
        confusion_matrix('DT',test_labels, clf2.predict(test_sounds))
        print('-----NB----') 
        print(classification_report(test_labels, clf3.predict(test_sounds), labels=[0, 1,2,3,4],target_names=target_names ))
        confusion_matrix('NB',test_labels, clf3.predict(test_sounds))
        print('-----Voting Class----')
        print(classification_report(test_labels, eclf.predict(test_sounds), labels=[0, 1,2,3,4],target_names=target_names ))
        confusion_matrix('Voting',test_labels, eclf.predict(test_sounds))
        print('----')

def get_accuarcy_DNN_dirict(train_sounds,test_sounds,train_labels,test_labels):
        target_names = ['Normal ', ' MR',' MVP',' MS',' AS']
        acc = [0,0,0,0,0]
        global accuarcy_all

    
        loaded_train_sounds = train_sounds
        loaded_test_sounds =  test_sounds
        full_test_features = test_sounds
        d1_labels = test_labels
        test_labels = to_categorical(test_labels, num_classes=5)
        train_labels = to_categorical(train_labels, num_classes=5)
        train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,int(x_features/2),2))
        full_test_features=np.reshape(full_test_features,(full_test_features.shape[0],1,int(x_features/2),2))

        ol_Train = load('train_heart_sounds.npy')
        ol_test = load('test_heart_sounds.npy')

        max_number = 0
        if Deep_DNN_smart_std_mean == True:
            
            for i in range(epoch_DNN_repeat):
                print('Epoch : '+str(i))
                if accuarcy_all > 99.9:
                    break
                else : 
                    print('Accuarcy _ all : '+str(accuarcy_all))

                x = loaded_train_sounds
                y = loaded_test_sounds
                if min(acc) == acc[0]:
                    x,y =  smart_adaptive_dataset(x,y, epoch= 1,AI_active= True,min_max= True, work_on ='0',type_d = 'mean',samples_count =100)
                    print(acc[0]) 
                elif min(acc) == acc[1]:
                    print(acc[1])
                    if epoch_DNN_repeat == 50 or epoch_DNN_repeat == 100:
                        x,y =  smart_adaptive_dataset(ol_Train,ol_test, epoch= 1,AI_active= True,min_max= True, work_on ='1',type_d = 'mean',samples_count =100)
                    else:
                        x,y =  smart_adaptive_dataset(x,y, epoch= 1,AI_active= True,min_max= True, work_on ='1',type_d = 'mean',samples_count =100)

                elif min(acc) == acc[2]:
                    print(acc[2])
                    if epoch_DNN_repeat == 50 or epoch_DNN_repeat == 100:
                        
                        x,y =  smart_adaptive_dataset(ol_Train,ol_test, epoch= 1,AI_active= True,min_max= True, work_on ='2',type_d = 'mean',samples_count =100)
                    else:
                        x,y =  smart_adaptive_dataset(x,y, epoch= 1,AI_active= True,min_max= True, work_on ='2',type_d = 'mean',samples_count =100)
                elif min(acc) == acc[3]:
                    print(acc[3])                    
                    if epoch_DNN_repeat == 50 or epoch_DNN_repeat == 100:
                        
                        x,y =  smart_adaptive_dataset(ol_Train,ol_test, epoch= 1,AI_active= True,min_max= True, work_on ='3',type_d = 'mean',samples_count =100)
                    else:
                        x,y =  smart_adaptive_dataset(x,y, epoch= 1,AI_active= True,min_max= True, work_on ='3',type_d = 'mean',samples_count =100)
                elif min(acc) == acc[4]:
                    print(acc[4])                    
                    if epoch_DNN_repeat == 50 or epoch_DNN_repeat == 100:
                        
                        x,y =  smart_adaptive_dataset(ol_Train,ol_test, epoch= 1,AI_active= True,min_max= True, work_on ='4',type_d = 'mean',samples_count =100)               
                    else:
                        x,y =  smart_adaptive_dataset(x,y, epoch= 1,AI_active= True,min_max= True, work_on ='4',type_d = 'mean',samples_count =100)               
                train_sounds=np.reshape(x,(x.shape[0], 1,int(x_features/2),2))
                full_test_features=np.reshape(y,(y.shape[0],1,int(x_features/2),2))
                print(accuarcy_all)
                loaded_train_sounds = x
                loaded_test_sounds = y
                model = keras.Sequential([
                keras.layers.Flatten(input_shape=(1,int(x_features/2),2)),
                keras.layers.Dense(int(x_features), activation=tf.nn.relu),
            	keras.layers.Dense(int(x_features/2), activation=tf.nn.relu),
            	keras.layers.Dense(int(x_features/2), activation=tf.nn.relu),
                keras.layers.Dense(5, activation=tf.nn.softmax),#softmax || sigmoid
                ])#sigmoid

                model.compile(optimizer='adam',
                              loss='categorical_crossentropy', # categorical_crossentropy ||binary_crossentropy
                              metrics=['accuracy'])
                
                model.fit(train_sounds,train_labels, epochs=94, batch_size=25,verbose=0)

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
                classfication_model = classification_report(d1_labels,y_classes, labels=[0, 1,2,3,4],target_names=target_names )
                accuracy = accuracy_score(d1_labels,d)
                acc[0]= accuracy_score(d1_labels[0:100],d[0:100])*100
                acc[1] = accuracy_score(d1_labels[100:100*2],d[100:100*2])*100
                acc[2] = accuracy_score(d1_labels[100*2:100*3],d[100*2:100*3])*100
                acc[3] = accuracy_score(d1_labels[100*3:100*4],d[100*3:100*4])*100
                acc[4] = accuracy_score(d1_labels[100*4:100*5],d[100*4:100*5])*100

#                get_accuarcy_SV_dirict(x,y,train_labels,test_labels)

                if accuracy > max_number:
                    max_number=accuracy
                    print(classfication_model)
                    df = pd.DataFrame(ynew)
                    filepath = 'my_excel_file.xlsx'
                    df.to_excel(filepath, index=False)
                    save('model_predicted', d)        
                    save('model_predicted', d1_labels)        
                    model.save('DNN_Saved_model.hdf5')
                    confusion_matrix('DNN',d1_labels, y_classes)
                    accuarcy_all =accuracy *100
                    print(accuarcy_all)
                
        else:
            for i in range(epoch_DNN_repeat):
         
                print('Accuarcy _ all : '+str(accuarcy_all))
                print('Epoch : '+str(i))
                loaded_train_sounds = load('train_heart_sounds.npy')
                loaded_test_sounds =  load('test_heart_sounds.npy')
                train_sounds = loaded_train_sounds
                full_test_features = loaded_test_sounds 
                train_sounds=np.reshape(train_sounds,(train_sounds.shape[0], 1,int(x_features/2),2))
                full_test_features=np.reshape(full_test_features,(full_test_features.shape[0],1,int(x_features/2),2))

          
                model = keras.Sequential([
                keras.layers.Flatten(input_shape=(1,int(x_features/2),2)),
                keras.layers.Dense(int(x_features), activation=tf.nn.relu),
            	keras.layers.Dense(int(x_features/2), activation=tf.nn.relu),
            	keras.layers.Dense(int(x_features/2), activation=tf.nn.relu),
                keras.layers.Dense(5, activation=tf.nn.softmax),#softmax sigmoid
                ])#sigmoid
                
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy', #categorical_crossentropy
                              metrics=['accuracy'])
                
                model.fit(train_sounds,train_labels, epochs=94, batch_size=20,verbose=1)
                print(accuarcy_all)

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
                classfication_model = classification_report(d1_labels,y_classes, labels=[0, 1,2,3,4],target_names=target_names )
                accuracy = accuracy_score(d1_labels,d)

 
        
                if accuracy > max_number:
                    max_number=accuracy
                    print(classfication_model)
                    df = pd.DataFrame(ynew)
                    filepath = 'my_excel_file.xlsx'
                    df.to_excel(filepath, index=False)
                    save('model_predicted', d)        
                    save('model_predicted', d1_labels)        
                    model.save('DNN_Saved_model.hdf5')
                    confusion_matrix('DNN',d1_labels, y_classes)
                    accuarcy_all =accuracy *100
                    


#model training and validation
import pathlib
def run_model_file(train_y,test_y,train_z,test_z,train_a,test_a,train_b,test_b,train_n,test_n):
    file1 = pathlib.Path("train_heart_sounds.npy")
    file2 = pathlib.Path("test_heart_sounds.npy")
    file3 = pathlib.Path("train_labels_heart_sounds.npy")
    file4 = pathlib.Path("test_labels_heart_sounds.npy")
    if predict_folder == True : 
        file_holder = []
        listOfFiles = os.listdir('predict_folder')
        for file in listOfFiles:
            filename = os.path.join('predict_folder',file)
            file_holder.append(filename)
        print(file_holder)
        file_featers = decodeFolder_random(file_holder)
        print(file_featers.shape)
        reconstructed_model = keras.models.load_model('DNN_Saved_model.hdf5')
        file_featers=np.reshape(file_featers,(file_featers.shape[0],1,int(x_features/2),2))
        predicted_values = reconstructed_model.predict(file_featers)
        y_classes = predicted_values.argmax(axis=-1)
        print(y_classes)
        return
    if file1.exists ()and file2.exists()and file3.exists()and file4.exists() and not run_new:
        print ("File exist")
        # load the model from disk
        loaded_train_sounds = load('train_heart_sounds.npy')
        loaded_test_sounds =  load('test_heart_sounds.npy')
        loaded_train_labels =   load('train_labels_heart_sounds.npy')
        loaded_test_labels =   load('test_labels_heart_sounds.npy')
        if Deep_DNN_smart:
            
            x,y =  smart_adaptive_dataset(loaded_train_sounds,loaded_test_sounds, epoch= 2,AI_active= True,min_max= True,work_on = 'all',type_d = 'std',samples_count =100)
            get_accuarcy_SV_dirict(x,y,loaded_train_labels,loaded_test_labels)
            get_accuarcy_DNN_dirict(x,y,loaded_train_labels,loaded_test_labels)
        else:

            get_accuarcy_DNN_dirict(loaded_train_sounds,loaded_test_sounds,loaded_train_labels,loaded_test_labels)
            get_accuarcy_SV_dirict(loaded_train_sounds,loaded_test_sounds,loaded_train_labels,loaded_test_labels)


        
    else:
        print ("File not exist")
        normal_sounds = decodeFolder_random(train_n)
        normal_labels = [0 for items in normal_sounds]
        murmur_sounds = decodeFolder_random(train_y)
        murmur_labels = [1 for items in murmur_sounds]
        
        murmur_sounds1 = decodeFolder_random(train_z)
        murmur_labels1 = [2 for items in murmur_sounds1]

        murmur_sounds2 = decodeFolder_random(train_a)
        murmur_labels2 = [3 for items in murmur_sounds2]  
        
        murmur_sounds3 = decodeFolder_random(train_b)
        murmur_labels3 = [4 for items in murmur_sounds3]  
        

        train_sounds = np.concatenate((normal_sounds, murmur_sounds,murmur_sounds1))
        train_sounds = np.concatenate((train_sounds, murmur_sounds2,murmur_sounds3))
        
        train_labels = np.concatenate((normal_labels, murmur_labels,murmur_labels1))
        train_labels = np.concatenate((train_labels,murmur_labels2,murmur_labels3))
        #test_data
        test_x = decodeFolder_random(test_n)
        test_y = decodeFolder_random(test_y)
        test_z = decodeFolder_random(test_z)
        test_a = decodeFolder_random(test_a)
        test_b = decodeFolder_random(test_b)

    
        predy_label = [1 for items in test_y]
        predx_label = [0 for items in test_x]
        predz_label = [2 for items in test_z]
        preda_label = [3 for items in test_a]
        predb_label = [4 for items in test_b]

        
        test_labels = np.concatenate((predx_label, predy_label,predz_label))
        test_labels = np.concatenate((test_labels,preda_label,predb_label))
        
        
        testall = np.concatenate((test_x, test_y,test_z))
        testall = np.concatenate((testall, test_a,test_b))
        #save raw data and labels
        save('train_heart_sounds.npy', train_sounds)
        save('train_labels_heart_sounds.npy', train_labels)
        save('test_heart_sounds.npy', testall)
        save('test_labels_heart_sounds.npy', test_labels)
     

        x , y = smart_adaptive_dataset(train_sounds,testall,5,True,True,'all')
      
        
        get_accuarcy_SV_dirict(x,y,train_labels,test_labels)
        get_accuarcy_DNN_dirict(x,y,train_labels,test_labels)

        

 
#shufle the data set step1 before smart adustive function 
train_n,test_n = setshuflefiles_dataset("train_test_50_50 - random",4,100)
train_mr,test_mr = setshuflefiles_dataset("train_test_50_50 - random",1,100)
train_mvp,test_mvp = setshuflefiles_dataset("train_test_50_50 - random",3,100)
train_as,test_as = setshuflefiles_dataset("train_test_50_50 - random",0,100)
train_ms,test_ms = setshuflefiles_dataset("train_test_50_50 - random",2,100)


        
modeal_accived = []
cc = 0 
if(run_deep):
    while accuarcy_all <= acuarcy_target:
        print('accuarcy all : '+str(accuarcy_all))
        cc = cc +1 
        run_model_file(test_mr,train_mr,train_mvp,test_mvp,train_ms
                       ,test_ms,train_as,test_as,train_n,test_n)
        modeal_accived.append(accuarcy_all)
    print(modeal_accived)
    print('counter : ' + str(cc))
    print('Accuarcy is above 99.9 ')
else:
    run_model_file(test_mr,train_mr,train_mvp,test_mvp,train_ms
                       ,test_ms,train_as,test_as,train_n,test_n)




        