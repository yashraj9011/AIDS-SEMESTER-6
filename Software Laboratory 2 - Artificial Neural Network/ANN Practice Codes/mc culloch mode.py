#!/usr/bin/env python
# coding: utf-8

# In[165]:


import numpy as np
np.random.seed(seed=2)
I = np.random.choice([1,4], 2) # take two random vectors between 0.1 and 1.3 as input
W = np.random.choice([4,6], 2) # take two random vectors between 4 and 6 as weight
print(f'Input vector:{I}, Weight vector:{W}')


# In[166]:


x = I @ W
print(f'Dot product: {x}')


# In[167]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[168]:


Y=sigmoid(x)
Y


# In[169]:


def linear_threshold_gate(Y :int, T: int):
    
    if Y >= T:
        print('Class A')
        W = np.random.choice([4,6], 2) # take two random vectors between 4 and 6 as weight
        W=(W+I).astype(np.int64)
        print(W)
        return 1
    else:
        print('Class B')
        W = np.random.choice([4,6], 2) # take two random vectors between 4 and 6 as weight
        W=(W-I).astype(np.int64)
        print(W)
        return 0


# In[170]:


T = 0.22
activation = linear_threshold_gate(Y, T)
print(f'Activation: {activation}')


# In[171]:


T = 4
activation = linear_threshold_gate(Y, T)
print(f'Activation: {activation}')


# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




