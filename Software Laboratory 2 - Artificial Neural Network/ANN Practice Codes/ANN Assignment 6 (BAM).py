#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd


# In[31]:


x1=[1,1,1,-1,-1,-1]
x2=[-1,1,-1,-1,-1,-1]
x3=[1,-1,1,1,-1,-1]
x4=[-1,1,1,-1,-1,-1]

wa=[3.1,2.1,1.1,1.3]
wb=[2.2,2.5,1.1,2.1]
wc=[3.1,1.4,2.2,2.8]
wd=[0.4,0.6,0.7,0.4]

outpt1 = x1[0]*wa[0] + x2[0]*wa[1] + x3[0]*wa[2]  + x4[0] *wa[3]

outpt2 = x1[1]*wb[0] + x2[1]*wb[1] + x3[1]*wb[2]  + x4[1] *wb[3]

outpt3 = x1[2]*wc[0] + x2[2]*wc[1] + x3[2]*wc[2]  + x4[2] *wc[3]

outpt4 = x1[3]*wd[0] + x2[3]*wd[1] + x3[3]*wd[2]  + x4[3] *wd[3]




print(outpt1)
print(outpt2)
print(outpt3)
print(outpt4)


# In[16]:


import numpy as np

class BAM:
    def __init__(self, x1, x2,x3,x4):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 =np.array(x3)
        self.x4 =np.array(x4)
        self.weights = np.zeros((len(self.x1)*len(self.x2), len(self.x3)*len(self.x4)))

        
    def forward(self, input_pattern):
        input_pattern = np.array(input_pattern)
        output1 = np.dot(input_pattern, self.weights.T)
        output2 = np.dot(output1, self.weights)
        return output2
    
    def retrieve(self, pattern):
        pattern = np.array(pattern).reshape(len(x1)*len(x2),len(self.x3)*len(self.x4))
        return np.dot(pattern, self.weights.T)
    


# In[17]:


x1=[1,1,1,-1,-1,-1]
x2=[-1,1,-1,-1,-1,-1]
x3=[1,-1,1,1,-1,-1]
x4=[-1,1,1,-1,-1,-1]


bam = BAM(x1,x2,x3,x4)

tar_pattern = [-1,1,1,1]
retrieved_pattern = bam.retrieve(tar_pattern)

print(retrieved_pattern)


# In[18]:


x1=[1,1,1,-1,-1,-1]
x2=[-1,1,-1,-1,-1,-1]
x3=[1,-1,1,1,-1,-1]
x4=[-1,1,1,-1,-1,-1]


bam = BAM(x1, x2, x3 , x4)

input_pattern = [-1,1,-1,1]
output = bam.forward(input_pattern)

print(output)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import numpy as np

class BAM:
    def __init__(self, x1, x2):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.weights = np.outer(self.x1, self.x2)
        
        
    def forward(self, input_pattern):
        input_pattern = np.array(input_pattern)
        output1 = np.dot(input_pattern, self.weights.T)
        output2 = np.dot(output1, self.weights)
        return output2
    
    def retrieve(self, pattern):
        pattern = np.array(pattern)
        return np.dot(pattern, self.weights.T)
    


# In[2]:


x1 = [1, -1, 1, -1]
x2 = [-1, 1, -1, 1]


bam = BAM(x1, x2)

new_pattern = [-1,1,1,1]
retrieved_pattern = bam.retrieve(new_pattern)

print(retrieved_pattern)


# In[3]:


x1 = [1, -1, 1, -1]
x2 = [-1, 1, -1, 1]

bam = BAM(x1, x2)

input_pattern = [-1,1,-1,1]
output = bam.forward(input_pattern)

print(output)


# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
print("We take two inputs patters, A and C")
print("\n")
input1=np.array([-1,1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1]).reshape(15,1)#coverting our 5*3 pattern into 15*1 for mathematical ease 
input2=np.array([-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1]).reshape(15,1)
input3=np.array([-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1]).reshape(15,1)#coverting our 5*3 pattern into 15*1 for mathematical ease 
input4=np.array([1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1]).reshape(15,1)
output1=np.array([-1,1,1]).reshape(1,3)
output4=np.array([-1,1,-1]).reshape(1,3)
output2=np.array([1,1,-1]).reshape(1,3)
output3=np.array([1,-1,1]).reshape(1,3)
print("The input for pattern A is")
print(input1)
print("\n")
print("The target for pattern A is")
print(output1)
print("\n")
print("The input for pattern C is")
print(input2)
print("\n")
print("The target for pattern C is")
print(output2)
print("\n")
print("\n")
print("\n")
inp_final=np.concatenate((input1,input2,input3,input4),axis=1)
out_final=np.concatenate((output1,output2,output3,output4),axis=0)
print("the initial weight for pattern A is:")
print(np.dot(input1,output1))
print("\n")
print("\n")
print("the initial weight for pattern C is:")
print(np.dot(input2,output2))
print("\n")
print("\n")
print("the final weights for pattern A and C is:")
weight=np.dot(inp_final,out_final)
print(weight)
print("\n")
print("\n")
print("\n")
print("\n")
print("Now for the testing phase\nWe multiply the input pattern with the weight matrix calculated above")
print("\nTesting for input pattern A\n")
print(input1.T,"*",weight)
y=np.dot(input1.T,weight)
y[y<0]=-1
y[y>=0]=1
print("y=",y)
print("\n")
print("\nTesting for input pattern C\n")
print(input2.T,"*",weight)
y=np.dot(input2.T,weight)
y[y<0]=-1
y[y>=0]=1
print("y=",y)
print("Since testing input for pattern A and C gives correct target values, testing is successful")
print("\n")
print("\n")
print("\nTesting for output Target A\n")
print(output1,"*",weight.T)
y=np.dot(output1,weight.T)
y[y<0]=-1
y[y>=0]=1
print("y=",y)
print("\n")
print("\nTesting for output Target C\n")
print(output2,"*",weight.T)
y=np.dot(output2,weight.T)
y[y<0]=-1
y[y>=0]=1
print("y=",y)

print("Since testing targets for pattern A and C gives correct input values, testing is successful")


# In[ ]:




