# nlp-PyTorch
神经网络工程包
# pytorch_Lstm.py
~~~~
# Author: Robert Guthrie
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
lstm=torch.nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
torch.manual_seed(1)
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
~~~~
# 运行结果
tensor([[[-0.0069,  0.0047,  0.3768]],  

        [[-0.0713,  0.0829,  0.1027]],  

        [[-0.0010, -0.0897, -0.0720]],  

        [[-0.0380, -0.0021, -0.0923]],  

        [[-0.0820,  0.0991, -0.1158]]], grad_fn=<MkldnnRnnLayerBackward0>)   
(tensor([[[-0.0820,  0.0991, -0.1158]]], grad_fn=<StackBackward0>), tensor([[[-0.2937,  0.2913, -0.2318]]], grad_fn=<StackBackward0>))  
