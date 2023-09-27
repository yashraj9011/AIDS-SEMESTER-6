#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
np.random.seed(seed=2)
I = np.random.choice([0.1,1.3], 3)# generate random vector I
W = np.random.choice([1.2,1.8], 3) # generate random vector W
print(f'Input vector:{I}, Weight vector:{W}')


# In[106]:


dot_product = I @ W
print(f'Dot product: {dot_product}')


# In[107]:


def linear_threshold_gate(dot :int, T: int):
    
    if dot >= T:
        print('Class A')
        return 1
    else:
        print('Class B')
        return 0


# In[108]:


T = 1
activation = linear_threshold_gate(dot, T)
print(f'Activation: {activation}')


# In[116]:


T = 2.1
activation = linear_threshold_gate(dot, T)
print(f'Activation: {activation}')


# In[ ]:





# In[ ]:




