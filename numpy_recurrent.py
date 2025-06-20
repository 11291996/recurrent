import numpy as np
#this is for getting my activation functions from activation repository
import sys
sys.path.append('../')
from activation.activation import activation

activation = activation()

np.random.seed(0)

#create rnn class
class rnn:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        #initialize weights
        self.U = np.random.rand(input_dim, hidden_dim)
        self.W = np.random.rand(hidden_dim, hidden_dim)
        self.V = np.random.rand(hidden_dim, output_dim)
        #add bias term
        self.b = np.random.rand(hidden_dim)
        
        #initialize hidden state
        self.h = np.zeros((hidden_dim,))
    
    def forward(self, x):
        #compute new hidden state
        self.h = activation.tanh(np.dot(x, self.U) + np.dot(self.h, self.W))
        
        #compute output
        y = activation.softmax(np.dot(self.h, self.V), std=True)
        
        return y

#set dimensions for rnn 
input_dim = 3
hidden_dim = 5
output_dim = 2
#initialize rnn
rnn = rnn(input_dim, hidden_dim, output_dim)
#example input
apple = np.array([1, 0, 0])
banana = np.array([0, 1, 0])
cherry = np.array([0, 0, 1])

#print the variable name 
dictionary = {'apple': apple, 'banana': banana, 'cherry': cherry}

#run the rnn model on the example input
for fruit, vector in dictionary.items():
    output = rnn.forward(vector)
    print(f"current word: {fruit}, output: {output}")