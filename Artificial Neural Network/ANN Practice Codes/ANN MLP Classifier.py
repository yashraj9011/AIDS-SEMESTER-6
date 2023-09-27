#!/usr/bin/env python
# coding: utf-8

# In[39]:


from sklearn.neural_network import MLPClassifier


# In[46]:


x=([1,2,5],[3,5,7])
y=[11,10]

clf=MLPClassifier(solver='adam',alpha=0.001,hidden_layer_sizes=(5,2),random_state=1)
clf.fit(x,y)



#solver lbfgs...sgd on gradient ..adam , choose activn funct
#TP TN FT FN for classification

#70% 30% pattern
#lambda is sttepness 
#power_t threshold value
#


# In[47]:


clf.predict(([1,2,1],[3,5,7]))


# In[ ]:





# In[ ]:





# In[ ]:




