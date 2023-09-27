#!/usr/bin/env python
# coding: utf-8

# # Name: Yashraj Deepak Devrat
# # Rollno: 31
# # Assignment no 1 :Activation Functions
# 

# ###  1.Write a Python program to plot a few activation functions that are being used in neural networks.

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[4]:


import math
from scipy.misc import derivative


# # 1.Identity Function 

# In[5]:


x=np.arange(-9,9)
y=x  #identity fun y=f(x)
plt.plot(x,y,c="red")
plt.title("Identity")
plt.scatter(x,y)
plt.grid()
plt.show


# # 2. Binary Function 

# In[6]:


#Binary function y=f(x)
x=np.arange(-10,10)
y=[]
for i in range (x.shape[0]):
    if (x[i]>0):
        y.append(1)
    else:
        y.append(0)
plt.plot(x,y,c="yellow")
plt.title("Binary")
plt.scatter(x,y)
plt.grid()
plt.show()


# # 3.Bipolar Function

# In[7]:


x=np.arange(-10,10)
y=[]
for i in range (x.shape[0]):
    if (x[i]>0):
        y.append(2)
    else:
        y.append(-2)
plt.plot(x,y,c="red")
plt.title("Bipolar")
plt.grid()
plt.scatter(x,y)
plt.hlines(y=0,xmin=-9,xmax=9)
plt.show()


# # 4. Sigmoid Function 

# In[8]:


x=np.arange(-10,10)
y=[]
for i in range (x.shape[0]):
   y.append((1/(1+np.exp(-x[i]))))
plt.plot(x,y,c="red")
plt.title("Sigmoid")
plt.grid()
plt.show()


# In[9]:


def sigmoid_function(x):
    z=1/(1+np.exp(-x))
    return z


# In[10]:


sigmoid_function(0.6)
print(x)


# In[13]:


import numpy as np
from matplotlib import pyplot as plt


def sig(x):
  return 1/(1+np.exp(-x))


def dsig(x):
  return sig(x) * (1- sig(x))


x_data = np.linspace(-6,6,100)
y_data = sig(x_data)
dy_data = dsig(x_data)


plt.plot(x_data, y_data, x_data, dy_data)
plt.title('Sigmoid Function & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


# # 2. Tanh function
# 

# In[12]:


import numpy as np
from matplotlib import pyplot as plt

# Tanh
def tanh_function(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# Tanh derivative
def tanh_deriv(z):
    return 1 - np.power(tanh_function(z),2)

# Generating data to plot
x_data = np.linspace(-6,6,100)
y_data = tanh_function(x_data)
dy_data = tanh_deriv(x_data)
 
# Plotting
plt.plot(x_data, y_data, x_data,dy_data)
plt.title('Tanh Function & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


# # 3.ReLU function

# In[13]:


import numpy as np
from matplotlib import pyplot as plt

# Relu function
def ReLU(x):
    return np.maximum(0.,x)
# Relu  derivative
def ReLU_grad(x):
    return np.greater(x, 0.).astype(np.float32)

# Generating data to plot
x_data = np.linspace(-6,6,100)
y_data = ReLU(x_data)
dy_data = ReLU_grad(x_data)
 
# Plotting
plt.plot(x_data, y_data, x_data,dy_data)
plt.title('ReLu Function & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


# # 4. Leaky ReLU

# In[14]:


import numpy as np
from matplotlib import pyplot as plt

# leaky Relu function
def leaky_ReLU(x):
    data = [max(0.05*value,value) for value in x]
    return np.array(data, dtype=float)
# leaky Relu  derivative
def der_leaky_ReLU(x):
  data = [1 if value>0 else 0.05 for value in x]
  return np.array(data, dtype=float)
  
# Generating data to plot
x_data = np.linspace(-6,6,100)
y_data = leaky_ReLU(x_data)
dy_data = der_leaky_ReLU(x_data)
 
# Plotting
plt.plot(x_data, y_data,x_data,dy_data)
plt.title('leaky ReLu Function & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


# # 5 Softmax Function

# In[15]:


import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x)
    return e_x / e_x.sum()

x = np.array([1, 0, 3, 5])
y = softmax(x)
y, x / x.sum()


# Generating data to plot
x_data = np.linspace(-6,6,100)
y_data = softmax(x_data)
#dy_data = tanh_prime_function(x_data)
 
# Plotting
plt.plot(x_data, y_data)
plt.title('Softmax Function')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


# #  6 Swish Function

# In[1]:


import numpy as np
from matplotlib import pyplot as plt

# Swish function
def swish(x):
    return x * 1/(1+np.exp(-x))

# Swish derivative  
def dswish(x):
    return  x * 1/(1+np.exp(-x)) + 1/(1+np.exp(-x)) * (1-x * 1/(1+np.exp(-x)))

# Generating data to plot
x_data = np.linspace(-8,8,50)
y_data = swish(x_data)
dy_data = dswish(x_data)

# Plotting
plt.plot(x_data, y_data,dy_data)
plt.title('Swish Function & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


# # 7 Exponential Function

# In[ ]:





# In[ ]:





# In[ ]:




