#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# Define the McCulloch-Pitts neural network
class McCullochPitts:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0
    
    def predict(self, inputs):
        linear_combination = np.dot(self.weights, inputs) + self.bias
        return 1 if linear_combination >= 0 else 0

# Define the inputs and expected outputs for the ANDNOT function
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([0, 0, 1, 0])

# Create a McCulloch-Pitts neural network with 2 inputs
neural_network = McCullochPitts(2)

# Train the neural network using the perceptron learning rule
learning_rate = 0.1
epochs = 10
for epoch in range(epochs):
    for input, expected_output in zip(inputs, expected_outputs):
        prediction = neural_network.predict(input)
        error = expected_output - prediction
        neural_network.weights += learning_rate * error * input
        neural_network.bias += learning_rate * error

# Test the neural network on the inputs
for input in inputs:
    print(f"Inputs: {input}")
    if neural_network.predict(input) == 1:
        print("ANDNOT output: True")
    else:
        print("ANDNOT output: False")


# In[ ]:




