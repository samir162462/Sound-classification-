# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 07:16:15 2020

@author: sam
"""
import random
import os
import numpy as np
def setshuflefiles_dataset(category,num,type_n):#num the catagory -> type num : frm where to whwere 

	music_files=[]
	local_files=[]
	counter = 0
	listOfFiles = os.listdir(category)
	for file in listOfFiles:
            counter = counter +1
            filename = os.path.join(category,file)
            local_files = os.listdir(filename)
            for i in range(len(local_files)):
                local_files[i] = os.path.join(filename,local_files[i])
            music_files.append(local_files)
            if counter == 6:
                break
	random.shuffle(music_files[num])


	return  music_files[num][0:type_n],music_files[num][type_n:200]

    

    
train,test = setshuflefiles_dataset("train_test_50_50 - random",0,100)
print(np.arange(len(train))[train==test])


            
