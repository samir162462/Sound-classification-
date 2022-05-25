# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 07:14:56 2020

@author: sam
"""


import numpy as np
from numpy import load
import random
import math 
import matplotlib.pyplot as plt


def loss_func(keyx,valuex,keyy,valuey):
    count_diff = 0
    fm = findMissing(keyx,keyy)
    count_diff = count_diff- len(fm)
    if len(keyx) > len(keyy):
        for i in range(len(keyx)):
            for j in range(len(keyy)):
                if keyx[i] == keyy[j]:
                    if valuex[i]>valuey[j]:
                        count_diff =count_diff+ 0
                    else:
                        count_diff =count_diff+ abs(valuex[i]-valuey[j])


    elif len(keyx) < len(keyy):
        for i in range(len(keyy)):
            for j in range(len(keyx)):
                if keyy[i] == keyx[j]:
                    if valuex[j]>valuey[i]:
                        count_diff =count_diff+ 0
                    else:
                        count_diff =count_diff+ abs(valuex[j]-valuey[i])
    else:
        for i in range(len(keyy)):
            for j in range(len(keyx)):
                if keyy[i] == keyx[j]:
                    if valuex[j]>valuey[i]:
                        count_diff =count_diff+ 0
                    else:
                        count_diff =count_diff+ abs(valuex[j]-valuey[i])
    if count_diff > 0:
        print(count_diff)
        return count_diff/200
    else:
        return 0


def print_freq(keyxx,valuexx,keyyy,valueyy):
    for i in range(len(keyxx)):
        if len(keyxx) > len(keyyy):
            try:
                print ("% d : % d  :-:  % d : % d"%(keyxx[i], valuexx[i],keyyy[i],valueyy[i]))
            except:
                print ("% d : % d  :-:  % d : % d"%(keyxx[i], valuexx[i],0,0))
        else:
            try:
                print ("% d : % d  :-:  % d : % d"%(keyxx[i], valuexx[i],keyyy[i],valueyy[i]))
            except:
                print ("% d : % d  :-:  % d : % d"%(0, 0,keyyy[i],valueyy[i]))
                
def plot_freq(keyxx,valuexx,keyyy,valueyy,p):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Scatter the data
    plt.bar(list(keyxx), valuexx, color='b', label="Train ")
    plt.bar(list(keyyy), valueyy, color='g', label="Test")
    plt.plot(keyxx,valuexx,'bs',label="Train ")
    plt.plot(keyyy,valueyy,'g^',label="Test ")
    plt.xlabel('Key')
    plt.ylabel('Frequancy')
    plt.title('Histogram: Training set and testing set  of class : '+str(p))
    plt.legend(loc="upper left")
    
#    plt.plot(keyyy,valueyy)
    plt.show()
    

def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
  
#    for key, value in freq.items(): 
#        print ("% d : % d"%(key, value))
    return freq

class sample_info:
  def __init__(self, key,diract,value,priority):
    self.key = key
    self.diract = diract
    self.value = value
    self.priority = priority
def findMissing(a, b): 
      
    # Store all elements of second  
    # array in a hash table 
    s = dict() 
    for i in range(len(b)): 
        s[b[i]] = 1
  
    # Print all elements of first array  
    # that are not present in hash table 
    x = []
    for i in range(len(a)): 
        if a[i] not in s.keys(): 
            x.append( a[i]) 
    return x
def trail_replace(l_x ,l_y,p,epoch,AI_active,min_max,type_d,samples_count):
    xv = [] 
    yv = []

    counter = 0 
    if type_d == 'mean':
        for x,y in zip( l_x[p*samples_count:(1+p)*samples_count] ,l_y[p*samples_count:(1+p)*samples_count]):
            xv.append (round(np.mean(x)))
            yv.append(round(np.mean(y)))
    elif type_d == 'std':
        for x,y in zip( l_x[p*samples_count:(1+p)*samples_count] ,l_y[p*samples_count:(1+p)*samples_count]):
            xv.append (round(np.std(x)))
            yv.append(round(np.std(y)))
    
    for i in range(1):
            counter = counter +1



            s_x = np.array(xv)
            s_y = np.array(yv)

            freqx = CountFrequency(s_x)
            freqy = CountFrequency(s_y)
            
            top_values_x = []
            top_values_y = []
            key_x = []
            key_y = []
            value_x =[]
            value_y =[]
            
            for key, value in freqx.items():
                key_x.append(key)
                value_x.append(value)
                if value >= 25 : 
                    top_values_x.append(key)
                elif(value >= 20):
                    top_values_x.append(key)
            unique_y = []
            c = 0 
            for key, value in freqy.items():
                key_y.append(key)
                value_y.append(value)
                try:
                  if freqx.items[1][c]  in  freqy.items[1][c]:
                      unique_y.append(freqy.items[1][c])
                  c = c+1
                except:
                  c=c
                if value >= 20 : 
                    top_values_y.append(key)
                elif(value >= 15):
                    top_values_y.append(key)


            des_list_x = []
            x_mising = findMissing(key_x,key_y)
            for i in range(len(key_x)):
                for t in range(len(key_y)):
                    if(key_x[i] == key_y[t]):
                        if value_x[i]-value_y[t] > 0:
                            s = sample_info(np.where(s_x == key_x[i])[0][0:math.floor(abs(value_x[i]-((value_x[i]+value_y[t])/2)))],
                                            math.floor(abs(value_x[i]-((value_x[i]+value_y[t])/2))),key_x[i],1)
                            if math.floor(abs(value_x[i]-((value_x[i]+value_y[t])/2))) >0:
                                des_list_x.append(s)
                for m in x_mising:
                    if(key_x[i] == m):
                        s = sample_info(np.where(s_x == m)[0][0:math.floor(value_x[i]/2)],math.floor(value_x[i]/2),key_x[i],2)
                        if(math.floor(value_x[i]/2) > 0):
                            des_list_x.append(s)


            print('Class : '+str(p))
            print_freq(key_x,value_x,key_y,value_y)
            plot_freq(key_x,value_x,key_y,value_y,p)

            des_list_y = []
                       
            y_mising = findMissing(key_y,key_x)

            for i in range(len(key_y)):
                for t in range(len(key_x)):
                    if(key_y[i] == key_x[t]):
                        if value_y[i]-value_x[t] > 0:
                            if max(value_y) == value_y[i]:
                                s = sample_info(np.where(s_y == key_y[i])[0][0:math.ceil(abs(value_y[i]-((value_y[i]+value_x[t])/2)))],
                                            math.ceil(abs(value_y[i]-((value_y[i]+value_x[t])/2))),key_y[i],0)
                            else:
                                s = sample_info(np.where(s_y == key_y[i])[0][0:math.ceil(abs(value_y[i]-((value_y[i]+value_x[t])/2)))],
                                            math.ceil(abs(value_y[i]-((value_y[i]+value_x[t])/2))),key_y[i],1)
                            des_list_y.append(s)

                for m in y_mising:
                    if(key_y[i] == m):
                        s = sample_info(np.where(s_y == m)[0][0:math.ceil(value_y[i]/2)],math.ceil(value_y[i]/2),key_y[i],2)
                        if(math.ceil(value_y[i]/2) == 1):
                            des_list_y.append(s)
                        elif(math.ceil(value_y[i]/2) > 1):
                            des_list_y.append(s)



            print('___Train\___')

            if type_d == 'mean':
                
                for i in des_list_x:
                    print('Indexs : '+str(i.key) +' Mean ->  '+ str(i.value)+' : count = '+str(i.diract)+' : proiority : '+str(i.priority) )
                print('___TEST\___')
                for i in des_list_y:
                    print('Indexs : '+str(i.key) +' Mean ->  '+ str(i.value)+' : count = '+str(i.diract)+' : proiority : '+str(i.priority) )    
                print(y_mising)
            elif type_d == 'std':
                for i in des_list_x:
                   print('Indexs : '+str(i.key) +' STD ->  '+ str(i.value)+' : count = '+str(i.diract)+' : proiority : '+str(i.priority) )
                print('___TEST\___')
                for i in des_list_y:
                    print('Indexs : '+str(i.key) +' STD ->  '+ str(i.value)+' : count = '+str(i.diract)+' : proiority : '+str(i.priority) )    
                print(y_mising)
            
            pro_x = []
            pro_y = []
            
            
            for i in des_list_x: #  train 
                for k in i.key:
                    pro_x.append(k)
            counter = 0
            rand_y = []
            temp_keysy = des_list_y
            for i in temp_keysy: # test
                if i.priority ==2:
                    counter = counter +1
                    pro_y.append(i.key[0])
                    temp_keysy.remove(i)
            for i in temp_keysy: #  train 
                for k in i.key:
                    rand_y.append(k)
            
#            for i in range(abs(remining_y-))
            print('pro x : '+str(len(pro_x)))        
            print('pro y : '+str(len(pro_y)))        
            print('counter : '+str(counter))        
            print(len(rand_y)) 
            new_array = random.sample( rand_y, len(rand_y) )
            print('rand_y : '+str(len(new_array)))        
            print('rand_y  shape : '+str(len(new_array)))        

            if len(new_array) + len(pro_y) <= len(pro_x):
                pro_x = pro_x[0:len(new_array) + len(pro_y)]
            if len(pro_x)-counter>0:
                for i in range(abs(len(pro_x)-counter)):
                    pro_y.append(new_array[i])
            else:
                for i in range(len(pro_x)):
                    pro_y = pro_y[0:len(pro_x)]
                

            print(pro_x)
            print(pro_y)
            temp = []

            for i in range(len(pro_x)):
                if type_d == 'mean':
                    print('Redirict Test -> Train : '+str(np.mean(l_y[p*samples_count+pro_y[i]]))+' -> '+str(np.mean(l_x[p*samples_count+pro_x[i]])))
                elif type_d == 'std':
                    print('Redirict Test -> Train : '+str(np.std(l_y[p*samples_count+pro_y[i]]))+' -> '+str(np.std(l_x[p*samples_count+pro_x[i]])))


                temp =np.array(l_y[p*samples_count+pro_y[i]])
                t = temp.copy()
                l_y[p*samples_count+pro_y[i]] = l_x[p*samples_count+pro_x[i]]
                l_x[p*samples_count+pro_x[i]] = t
                
                xv = []
                yv = []
#                print(np.std(l_x[p*samples_count+pro_x[0]])) 

            for x,y in zip( l_x[p*samples_count:(1+p)*samples_count] ,l_y[p*samples_count:(1+p)*samples_count]):
                if type_d == 'mean':
                    xv.append (round(np.mean(x)))
                    yv.append(round(np.mean(y)))            
                elif type_d == 'std':
                    xv.append (round(np.std(x)))
                    yv.append(round(np.std(y)))    

                s_x = np.array(xv)
                s_y = np.array(yv)
    
    
            keyxx = []
            valuexx= [] 
            keyyy = []
            valueyy= [] 
                
            freqx = CountFrequency(s_x)
            for k , v in freqx.items():
                keyxx.append(k)
                valuexx.append(v)
            freqy = CountFrequency(s_y)
            for k , v in freqy.items():
                keyyy.append(k)
                valueyy.append(v)
            
            l_F = loss_func(keyxx,valuexx,keyyy,valueyy)
            print('loss : '+str(l_F))
#    print_freq(keyxx,valuexx,keyyy,valueyy)
    return l_x , l_y 
    
def smart_adaptive_dataset(train_sounds,test_sounds,epoch,AI_active,min_max,work_on,type_d,samples_count):
    
    loaded_train_sounds = train_sounds
    loaded_test_sounds = test_sounds
    


    print('-------')
    
    if work_on == 'all':
        x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,0,1,AI_active,min_max,type_d,samples_count)
        for i in range(epoch):
            x,y = trail_replace(x,y,1,1,AI_active,min_max,type_d,samples_count)
            x,y = trail_replace(x,y,2,1,AI_active,min_max,type_d,samples_count)
            x,y = trail_replace(x,y,3,1,AI_active,min_max,type_d,samples_count)
            x,y = trail_replace(x,y,4,1,AI_active,min_max,type_d,samples_count)
        return x,y 
    elif work_on == '0':   
        x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,0,1,AI_active,min_max,type_d,samples_count)
        for i in range(epoch-1):
            x,y = trail_replace(x,y,0,1,AI_active,min_max,type_d,samples_count)
        
        return x,y 
    elif work_on == '1':
        xx =[]
        yy =[]
        x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,1,1,AI_active,min_max,type_d,samples_count)
        if epoch-1 == 0:
            return x , y
        for i in range(epoch-1):
            x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,1,1,AI_active,min_max,type_d,samples_count)
            xx,yy = trail_replace(x,y,1,1,AI_active,min_max,type_d,samples_count)
        return xx,yy 

    elif work_on == '2':
        xx =[]
        yy =[]
        x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,2,1,AI_active,min_max,type_d,samples_count)
        if epoch-1 == 0:
            return x , y
        for i in range(epoch-1):
            xx,yy = trail_replace(x,y,2,1,AI_active,min_max,type_d,samples_count)
            print(len(xx))
        return xx,yy 

    elif work_on == '3':
        x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,3,1,AI_active,min_max,type_d,samples_count)
        if epoch-1 == 0:
            return x , y
        for i in range(epoch-1):
            x,y = trail_replace(x,y,3,1,AI_active,min_max,type_d,samples_count)
        return x,y 

    elif work_on == '4':
        x,y = trail_replace(loaded_train_sounds,loaded_test_sounds,4,1,AI_active,min_max,type_d,samples_count)
        if epoch-1 == 0:
            return x , y
        for i in range(epoch-1):
            x,y = trail_replace(x,y,4,1,AI_active,min_max,type_d,samples_count)
        return x,y 





    
    return x,y 
    
#    
loaded_train_sounds = load('train_heart_sounds.npy')
loaded_test_sounds =  load('test_heart_sounds.npy')
#tt,ss =  smart_adaptive_dataset(loaded_train_sounds,loaded_test_sounds, epoch= 2,AI_active= False,min_max= True,work_on = '1',type_d='std',samples_count=100)#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)
#tt,ss =  smart_adaptive_dataset(loaded_train_sounds,loaded_test_sounds, epoch= 2,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='std',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='std',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='std',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 1,AI_active= False,min_max= True,work_on = '1',type_d='std',samples_count=100)
#tt,ss =  smart_adaptive_dataset(tt,ss, epoch= 2,AI_active= False,min_max= True,work_on = '1',type_d='mean',samples_count=100)


    
    
    
    
    
    
    
    