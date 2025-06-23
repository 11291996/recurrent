import torch.nn as nn
import torch


# A simple recurrent neural network (RNN) model using PyTorch
#set dimensions for rnn 
input_dim = 3
hidden_dim = 5
output_dim = 2
#initialize rnn
rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False)
#pytroch rnn only has U and W, so V is needed 
V = nn.Linear(hidden_dim, output_dim)  # Output layer to map hidden state to output dimension
#example input
apple = torch.tensor([1, 0, 0], dtype=torch.float32)
banana = torch.tensor([0, 1, 0], dtype=torch.float32)
cherry = torch.tensor([0, 0, 1], dtype=torch.float32)
#print the variable name 
dictionary = {'apple': apple, 'banana': banana, 'cherry': cherry}

# Initialize the hidden state
h0 = torch.zeros(1, 1, hidden_dim)  # (num_layers, batch_size, hidden_size)

for name, vector in dictionary.items():
    # Reshape the input to match the expected input shape for RNN
    input_tensor = vector.view(1, 1, -1)  # (seq_len, batch_size, input_size)
    
    # Forward pass through the RNN
    out, hn = rnn(input_tensor, h0)  # out: (seq_len, batch_size, hidden_size)
    
    # Pass the last hidden state through the output layer
    output = V(out[-1])  # Use the last output from the RNN

    # Apply softmax to the output
    output = torch.softmax(output, dim=1)

    print(f"current word: {name}, output: {output.detach().numpy()}")
    h0 = hn  # Update the hidden state for the next iteration