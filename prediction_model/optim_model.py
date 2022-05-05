from torch.autograd import Variable
import torch

class Optim_Condition:
    def __init__(self, net, x_lb, x_ub, n_input, n_output) -> None:
        self.net = net
        self.x_lb, self.x_ub = x_lb, x_ub
        self.n_input, self.n_output = n_input, n_output
        self.F = torch.rand([n_input, n_output])

    def loss_func(self, size):
        self.x, self.b = self.sample(size)
        self.x = Variable(self.x, requires_grad = True)
        self.b_pred = self.net(self.x)
        # prediction_error  = torch.sum((self.b_pred - self.b)**2)
        loss = torch.nn.MSELoss()
        prediction_error = loss(self.b_pred, self.b)
        return prediction_error
    
    def sample(self, size):
        x = (self.x_ub-self.x_lb)*torch.rand([size, self.n_input]) + self.x_lb*torch.ones([size, self.n_input])
        return x, self.predict(x)
    
    def predict(self, x):
        self.b = torch.mul(torch.mm(x, self.F), torch.mm(x, self.F))
        # self.b = torch.mm(x, self.F)
        return self.b

