
from net import Net
from train import Train
import torch
from optim_model import Optim_Condition

n_input, n_output = 10, 2
net = Net(n_layer = 3, n_hidden = [60,50,40,30], n_input=n_input, n_output=n_output)
print(net)

kkt_condition = Optim_Condition(net, -10, 10, n_input, n_output)

train = Train(net, kkt_condition, BATCH_SIZE = 2**8)
train.train(epoch = 10**5, lr = 0.001)
train.save_model()
