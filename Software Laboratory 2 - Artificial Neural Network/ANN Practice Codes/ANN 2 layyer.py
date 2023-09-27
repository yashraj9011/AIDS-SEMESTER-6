#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


def sig(x,deriv=False):
    if(deriv==True):
        return x* (1-x)
    return 1/(1+np.exp(-x))


# In[3]:


#input dataset
X=np.array([  [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1]])

#output dataset

y=np.array([[0,0,1,1]]).T

#seed random for random distribution

np.random.seed(1)

#initialize weights with randomly with mean 0
#weight matrix 3 1
synapse0 = 2*np.random.random((3,1)) -1


# In[4]:


synapse0


# In[5]:


for i in range(1000):
    layr0=X   #feedforward to layr0
    layr1=sig(np.dot(layr0,synapse0))   #actvn fun 
    #calculte error
    
    layr1_error= y-layr1
    # mltiply error backpropagtd
    #slope of sigmoid at 0 and 1
    layr1_delta= layr1_error * sig(layr1,True)
    
    #update weights as per the errors backpropagation
    
    synapse0 += np.dot(layr0.T,layr1_delta)
    
    


# In[ ]:





# In[6]:


print("output after training")
print(layr1)
print("Actual output")
print(y)


# In[7]:


#input dataset
X=np.array([  [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1]])

#output dataset

y=np.array([[0,1,1,1]]).T

#seed random for random distribution

np.random.seed(1)

#initialize weights with randomly with mean 0
#weight matrix 3 1
synapse0 = 2*np.random.random((3,1)) -1


# In[8]:


for i in range(1000):
    layr0=X   #feedforward to layr0
    layr1=sig(np.dot(layr0,synapse0))   #actvn fun 
    #calculte error
    
    layr1_error= y-layr1
    # mltiply error backpropagtd
    #slope of sigmoid at 0 and 1
    layr1_delta= layr1_error * sig(layr1,True)
    
    #update weights as per the errors backpropagation
    
    synapse0 += np.dot(layr0.T,layr1_delta)
    
    


# In[9]:


layr1


# In[10]:


y


# In[11]:


#Split data to training and validation data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
#Weights
w0 = 2*np.random.random((4, 5)) - 1 #for input   - 4 inputs, 3 outputs
w1 = 2*np.random.random((5, 3)) - 1 #for layer 1 - 5 inputs, 3 outputs
#learning rate
n = 0.1


# In[12]:


for i in range (100000):

#Feed forward network

    layer0 = X_train
    layer1 = sigmoid_func(np.dot(layer0, w0))
    layer2 = sigmoid_func(np.dot(layer1, w1))
    #Back propagation using gradient descent
    layer2_error = y_train - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot (w1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
    w1 += layer1.T.dot(layer2_delta) * n
    w0 += layer0.T.dot(layer1_delta) * n
    error = np.mean(np.abs(layer2_error))
    errors.append(error)


# In[ ]:





# In[ ]:





# In[ ]:




