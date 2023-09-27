#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y


# In[2]:


a=np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],
           [1,1,0,0],[1,0,0,1],[1,0,1,0],[0,1,1,0],[0,0,1,1],
           [0,1,1,1],[1,0,1,1],[1,1,1,0],[1,1,0,1],[1,1,1,1]])


# In[3]:


a


# In[4]:


w=np.array=([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
w


# In[5]:


agg=np.dot(a,w) 
agg


# In[6]:


s2=agg[7]
s2


# In[7]:


y4=sigmoid(s2)
y4


# In[8]:


s1=np.dot(a,w) 


# In[9]:


s1


# 

# In[10]:


y2=sigmoid(3)
print(y2)


# In[11]:


s2=np.dot(a,w) [5]
s2


# In[12]:


s4=np.dot(a,w)  [3]
s4


# In[13]:


y1=sigmoid(s4)
print(y1)


# In[14]:


y2=sigmoid(s2)
y2


# In[ ]:





# In[2]:


import numpy as np

# Define the perceptron function
def perceptron(inputs, weights, bias):
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + bias
    # Apply the step function to the weighted sum
    if weighted_sum >= 0:
        return 1
    else:
        return 0

# Define the training data
data = {
    '48': 0, # '0' in ASCII
    '49': 1, # '1' in ASCII
    '50': 0, # '2' in ASCII
    '51': 1, # '3' in ASCII
    '52': 0, # '4' in ASCII
    '53': 1, # '5' in ASCII
    '54': 0, # '6' in ASCII
    '55': 1, # '7' in ASCII
    '56': 0, # '8' in ASCII
    '57': 1, # '9' in ASCII
}

# Initialize the weights and bias
weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # One weight for each ASCII code
bias = -5

inputs = np.array([int(c) for c in x]).reshape((1, 2))
weights = np.random.rand(len(data.keys()[0]),)




# Train the perceptron
for i in range(10000):
    for x, y in data.items():
        inputs = np.array([int(c) for c in x])
        output = perceptron(inputs, weights, bias)
        error = y - output
        weights += error * inputs
        bias += error

# Test the perceptron
while True:
    num = input('Enter a number (0-9): ')
    if num not in ['48', '49','50','51','52','53', '54', '55', '56', '57']:
        print('Invalid input.')
        continue
    inputs = np.array([int(c) for c in num])
    output = perceptron(inputs, weights, bias)
    if output == 1:
        print('Odd')
    else:
        print('Even')


# In[3]:


inputs = np.array([int(c) for c in x]).reshape((1, 2))
weights = np.random.rand(len(data.keys()[0]),)


# In[ ]:




