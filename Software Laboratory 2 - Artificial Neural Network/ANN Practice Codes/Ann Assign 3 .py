#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np

# Define the training data
training_data = np.array([[48, 0],  # '0' is even
                         [49, 1],  # '1' is odd
                         [50, 0],  # '2' is even
                         [51, 1],  # '3' is odd
                         [52, 0],  # '4' is even
                         [53, 1],  # '5' is odd
                         [54, 0],  # '6' is even
                         [55, 1],  # '7' is odd
                         [56, 0],  # '8' is even
                             [57, 1]]) # '9' is odd

# Separate the input features (ASCII values) and the target labels (even/odd)
X = training_data[:, 0].reshape(-1, 1)
y = training_data[:, 1]

# Create and train the perceptron
perceptron = np.where(np.sum(X, axis=1) % 2 == 0, 0, 1)

# Test the trained perceptron
test_data = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57])
predictions = np.where(np.sum(test_data.reshape(-1, 1), axis=1) % 2 == 0, 0, 1)

# Print the predictions
for ascii_val, prediction in zip(test_data, predictions):
    number = chr(ascii_val)
    parity = "even" if prediction == 0 else "odd"
    print(f"Number: {number}, ASCII value: {ascii_val} , Output: {parity}.")


# In[ ]:




