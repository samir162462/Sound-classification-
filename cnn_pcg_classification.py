# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:31:03 2020

@author: samir filfil
"""


from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os 


def decodeFolder(category):

	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	for file in listOfFiles:
		filename = os.path.join(category,file)
		
	return filename

test_X = decodeFolder("Full murmur train")
train_X = decodeFolder("Full Normal train")
trace_sound = decodeFolder("full trace")
test_Y = [0 for items in test_X]
train_Y = [1 for items in train_X]


print('Training data shape : ', train_X, train_Y)

print('Testing data shape : ', test_X, test_Y)



classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)




plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))