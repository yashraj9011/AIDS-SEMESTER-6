#!/usr/bin/env python
# coding: utf-8

# 
# # Name : Yashraj Deepak Devrat
# # Roll no : 31
# 

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.DataFrame([[8,8,0],[7,9,0],[6,10,1],[5,12,1],[7,9,0],[6,10,1]], columns=['cgpa', 'profile_score', 'placed'])


# In[4]:


df


# In[5]:


print(df.shape)


# In[6]:


sns.scatterplot(x=df['cgpa'],y=df['profile_score'],hue=df['placed'])


# In[7]:


X = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[8]:


from sklearn.linear_model import Perceptron
p = Perceptron()


# In[9]:


p.fit(X,y)


# In[10]:


p.coef_


# In[11]:


p.intercept_


# In[12]:


from mlxtend.plotting import plot_decision_regions


# In[13]:


plot_decision_regions(X.values, y.values, clf=p, legend=3)


# In[ ]:





# In[ ]:




