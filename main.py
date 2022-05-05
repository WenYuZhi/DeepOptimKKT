
from net import Net
from optim_model import Optim_Condition
from quad_demo import Quad_Demo
from train import Train
from test import Test
import torch

FILE_PATH = './trainning_data\\qp_sinx_nd=10_nc=0_in=12_out=10\\case2_50000\\'
n_dim, n_constrs = 10, 0
d_input, d_output, n_sample = 12, 10, 50000

quad_demo = Quad_Demo(n_dim = n_dim, d_input = d_input, d_output = d_output, n_constrs = n_constrs)
quad_demo.load_data(file_path=FILE_PATH, n_sample=n_sample, x_up=10, x_down=1)

'''
quad_demo.get_uncertainty(n_sample=n_sample, x_up=10, x_down=1)
quad_demo.transforms()
quad_demo.bulid_model()
quad_demo.solve()
quad_demo.save_data()
'''

quad_demo.validate_stationary()
quad_demo.split_train_test(rate = 0.1)

net_output = d_output + n_dim + n_constrs
net = Net(n_layer = 4, n_hidden = 30, d_input=d_input, d_output=net_output)
print(net)

# net = torch.load('net_model.pkl') 
kkt_eq = Optim_Condition(net, quad_demo)

train = Train(net, kkt_eq, BATCH_SIZE = 2**8)
train.train(epoch = 2*10**4, lr = 0.0001)
train.save_model()

'''
test = Test(net, quad_demo)
test.get_loss()
test.save_solution()
'''