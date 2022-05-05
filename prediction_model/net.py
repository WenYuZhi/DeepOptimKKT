
import torch
import torch.nn as nn 

class Net(nn.Module):
    # n_layer: the number of hidden layers
    # n_hidden: the number of vertices in each layer, its index from 0 to n_layer
    def __init__(self, n_layer: int, n_hidden: list, n_input: int, n_output: int):
        super(Net, self).__init__()
        self.n_input, self.n_output = n_input, n_output
        self.input_layer = nn.Linear(self.n_input, n_hidden[0])
        # self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) if i%2==0 else nn.BatchNorm1d(n_hidden) for i in range(n_layer)])
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden[i], n_hidden[i+1]) for i in range(n_layer)])
        self.output_layer = nn.Linear(n_hidden[n_layer], self.n_output)
        
    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out

    def act(self, x):
        # return x * torch.sigmoid(x)
        # return torch.sigmoid(x)
        # return torch.tanh(x)
        return torch.relu(x)