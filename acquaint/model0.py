import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearQNet(nn.Module):
    def __init__(self, in_size, hidden_layers, out_size):
        super().__init__()
        self.linear0 = nn.Linear(in_size, hidden_layers)                        # a neural network with input size in_size followed by a layer of size hidden_size
        self.linear1 = nn.Linear(hidden_layers, out_size)                       # a neural network with input size hidden_size followed by an output layer of size out_size
        
    def forward(self, in_data):
        '''performs a forward pass with in_data as input and uses the ReLU activation function'''
        x = F.relu(self.linear0(in_data))
        x = self.linear1(x)
        
        return x
    
def main():
    model = LinearQNet(3, 16, 2)
    in_data = torch.randn(1, 3)
    print(f'input data = {in_data}')
    output = model(in_data)
    print(f'output = {output}')
    
if __name__ == '__main__':
    main()