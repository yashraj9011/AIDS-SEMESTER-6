#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

# Define input data
X = np.array([[1.0, 2.0, 3.0]])

# Define weights and biases
W = np.array([[0.2, 0.5, 0.1]])
b = np.array([[0.1]])

# Perform forward propagation
z = np.dot(W, X.T) + b
y = 1.0 / (1.0 + np.exp(-Z))

# Print output
print("z =", Z)
print("y =", y)


# In[ ]:





# In[1]:


import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        
    def backward(self, X, y, learning_rate):
        # Backward pass
        delta3 = self.a2
        delta3[range(X.shape[0]), y] -= 1
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            
    def predict(self, X):
        # Predict class labels
        self.forward(X)
        return np.argmax(self.a2, axis=1)


# In[2]:


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)
model.train(X, y, learning_rate=0.1, num_epochs=10000)

predictions = model.predict(X)
print(predictions)


# In[ ]:




