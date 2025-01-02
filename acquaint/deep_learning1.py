import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Binary2DecimalNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear0 = nn.Linear(in_size, hidden_size)                        # a neural network with input size in_size and an output layer of size out_size
        self.linear1 = nn.Linear(hidden_size, out_size)
        
    def forward(self, in_data):
        '''performs a forward pass with in_data as input and uses the ReLU activation function'''
        x = F.relu(self.linear0(in_data))
        x = self.linear1(x)
        
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

def generateData(num_samples=10, num_bits=5):
    x = np.random.randint(2, size=(num_samples, num_bits))                      # generate a random binary string
    y = np.sum(x * 2 ** np.arange(5)[::-1], axis=1, dtype=np.float32)           # gets a sum of the binary string
    
    return torch.from_numpy(x).float(), torch.from_numpy(y).unsqueeze(1)
        
def main():
    in_size = 5
    hidden_layer_size = 4
    out_size = 1
    
    model = Binary2DecimalNN(in_size, hidden_layer_size, out_size)
    criterion = nn.MSELoss()                                           # Mean Squared Error loss
    optimiser = optim.Adam(model.parameters(), lr=.01)
    
    num_samples = 1000
    num_epochs = 1000
    
    # train the model
    for epochs in range(num_epochs):
        x_train, y_train = generateData(num_samples, 5)
        
        # forward pass
        outputs = model(x_train)
        # calculates the loss
        loss = criterion(outputs, y_train)
        
        # backward pass and optimisation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if (epochs + 1) % 100 == 0:
            print(f'Epoch [{epochs+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model.save()
    x_test, y_test = generateData()
    predicted = model(x_test)
    print('Predicted:')
    print(predicted.detach().numpy())
    print('Actual:')
    print(y_test.numpy())
    
if __name__ == '__main__':
    main()