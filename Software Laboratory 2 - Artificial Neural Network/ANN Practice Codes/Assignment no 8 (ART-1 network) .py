#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


#chatgpt


# In[4]:


class ART1NeuralNetwork:
    def __init__(self, num_input, vigilance):
        self.num_input = num_input
        self.vigilance = vigilance
        self.weights = np.zeros((1, num_input))
        self.bias = 0

    def reset(self):
        self.weights = np.zeros((1, self.num_input))
        self.bias = 0

    def feed_forward(self, input_vec):
        # Rescale input vector to binary values (0 or 1)
        input_vec = np.where(input_vec >= 0.5, 1, 0)

        # Calculate the choice function
        choice = np.dot(self.weights, input_vec) / (self.bias + np.sum(input_vec))

        # Check if the choice is above the vigilance parameter
        if choice >= self.vigilance:
            # Update the weights and bias
            self.weights = (self.weights + np.outer(choice, input_vec)) / (1 + np.sum(input_vec))
            self.bias = (self.bias + np.sum(input_vec)) / (1 + self.num_input)

        return choice


# In[14]:


nn = ART1NeuralNetwork(num_input=3, vigilance=0.5)
input_vec = np.array([1, 0.5, 0])
choice = nn.feed_forward(input_vec)


# In[ ]:





# In[15]:


rho=0.4
alpha = 2
m=3
b=0
n=4


# In[16]:


b=(1/1+n)
print(b)


# In[17]:


b=np.array([[0.2,0.2,0.2],[0.2,0.2,0.2],[0.2,0.2,0.2],[0.2,0.2,0.2]])


# In[18]:


b


# In[10]:


t=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
t


# In[11]:


x1=1
x2=0
x3=1
x4=0
b11=b21=b31=b41=b12=b22=b32=b42=0.2


# In[12]:


o1 = x1*b11 + x2*b21 + x3*b31 + x4*b41
o1


# In[19]:


o2= x1*b12 + x2*b22 + x3*b32 + x4*b42
o2


# In[ ]:





# In[ ]:




