#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# In[7]:


# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[8]:


len(X_train)


# In[9]:


len(X_test)


# In[11]:


X_train.shape


# In[12]:


X_train.shape[0]


# In[13]:


X_train[0].shape


# In[14]:


X_train[0]


# In[17]:


plt.matshow(X_train[4])


# In[ ]:





# In[ ]:





# In[20]:


# Define the model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# In[22]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# In[ ]:





# In[ ]:





# In[ ]:




