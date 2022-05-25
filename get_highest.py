# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:28:38 2020

@author: sam
"""
import numpy as np
import keras 


a = [
    [
     0.0,
     0.24320757858085434,
     0.14893361727523413,
     0.29786723455046826,
     0.18838778030301612,
     0.12160378929042717
    ],
    [
     0.23717478210768014,
     0.0,
     0.16770789675478251,
     0.20539938644228997,
     0.25981195646349819,
     0.1299059782317491
    ],
    [
     0.21681956134183847,
     0.250361664212574,
     0.0,
     0.23178986094050727,
     0.16390018248131957,
     0.13712873102376066
    ],
    [
     0.2933749527592357,
     0.20744741852633861,
     0.15681550844086434,
     0.0,
     0.18554661183269694,
     0.15681550844086434
    ]
    ]
    

for i in a:
    print(np.std(i))

print(np.corrcoef(a))






#result = np.where(a == np.amax(a))
# 
#print('Tuple of arrays returned : ', result)
# 
#print('List of coordinates of maximum value in Numpy array : ')
## zip the 2 arrays to get the exact coordinates
#listOfCordinates = list(zip(result[0], result[1]))
## travese over the list of cordinates
#for cord in listOfCordinates:
#    print(cord)
    
    
    
    
    
    
    
    
    
    