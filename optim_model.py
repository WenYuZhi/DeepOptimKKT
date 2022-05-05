import opcode
from torch.autograd import Variable
from quad_demo import Quad_Demo
import torch

class Optim_Condition:
    def __init__(self, net, quad_demo) -> None:
        self.net, self.quad_demo = net, quad_demo
        self.d_output, self.n_dim, self.n_sample = self.quad_demo.d_output, self.quad_demo.n_dim, self.quad_demo.n_sample
    
    def loss_func(self, size):
        self.x, b, w = self.quad_demo.sample(size)
        self.x = Variable(self.x, requires_grad = True)
        y_est = self.net(self.x)
        b_est, w_est = y_est[:,0:self.d_output], y_est[:,self.d_output:]
        stationarity_error = torch.mean(torch.abs(torch.addmm(input=b_est, mat1=w_est, mat2=self.quad_demo.A)))
        prediction_error  = torch.mean(torch.abs(b_est - b))
        optimum_error = torch.mean(torch.abs(w_est - w))
        stationarity_error = torch.sum((torch.addmm(input=b_est, mat1=w_est, mat2=self.quad_demo.A))**2)
        prediction_error  = torch.sum((b_est - b)**2)
        optimum_error = torch.sum((w_est - w)**2)
        
        self.x1 = self.quad_demo.random_sample(size)
        self.x1 = Variable(self.x1, requires_grad = True)
        y1_est = self.net(self.x1)
        b1_est, w1_est = y1_est[:,0:self.d_output], y1_est[:,self.d_output:]
        stationarity_error += torch.mean(torch.abs(torch.addmm(input=b1_est, mat1=w1_est, mat2=self.quad_demo.A)))
        stationarity_error += torch.sum((torch.addmm(input=b1_est, mat1=w1_est, mat2=self.quad_demo.A))**2)
        return prediction_error + optimum_error + stationarity_error
    
    def loss_func1(self, size):
        self.x, b, _ = self.quad_demo.sample(size)
        self.x = Variable(self.x, requires_grad = True)
        y_est = self.net(self.x)
        b_est = y_est[:,0:self.d_output]
        prediction_error  = torch.sum((b_est - b)**2)
        return prediction_error