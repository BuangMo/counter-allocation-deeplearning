import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Binary2DecimalNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Binary2DecimalNN, self).__init__()
        self.linear0 = nn.Linear(in_size, hidden_size)                        # a neural network with input size in_size and an output layer of size out_size
        self.linear1 = nn.Linear(hidden_size, out_size)
        
    def forward(self, in_data):
        '''performs a forward pass with in_data as input and uses the ReLU activation function'''
        x = F.relu(self.linear0(in_data))
        x = self.linear1(x)
        
        return x

def generateData(num_samples=10, num_bits=5):
    x = np.random.randint(2, size=(num_samples, num_bits))                      # generate a random binary string
    y = np.sum(x * 2 ** np.arange(5)[::-1], axis=1, dtype=np.float32)           # gets a sum of the binary string
    
    return torch.from_numpy(x).float(), torch.from_numpy(y).unsqueeze(1)

def main():
    in_size = 5
    hidden_layer_size = 16
    out_size = 1
    
    # load the model (requires the exact model definition)
    model = Binary2DecimalNN(in_size, hidden_layer_size, out_size)
    model.load_state_dict(torch.load('Model/model.pth'))
    
    # set the model to evaluation mode
    model.eval()
    
    # test the model
    x_test, y_test = generateData()
    predicted = model(x_test)
    print('Predicted:')
    print(predicted.detach().numpy())
    print('Actual:')
    print(y_test.numpy())
    
if __name__ == '__main__':
    main()