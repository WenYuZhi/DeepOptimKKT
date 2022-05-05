import torch
import numpy as np
import pandas as pd

class Test():
    def __init__(self, net, quad_demo):
        self.net, self.quad_demo = net, quad_demo
        self.n_dim, self.n_sample = self.quad_demo.n_dim, self.quad_demo.n_sample
    
    def get_loss(self):
        self.b_est, self.w_est = self.predict(self.quad_demo.x)
        self.b_error = torch.norm(self.quad_demo.b - self.b_est) / self.b_est.shape[0]
        self.w_error = torch.norm(self.quad_demo.w - self.w_est) / self.w_est.shape[0]
        print("the train MSE of b: {}".format(self.b_error))
        print("the train MSE of w: {}".format(self.w_error))

        self.b_est_test, self.w_est_test = self.predict(self.quad_demo.x_test)
        self.b_error_test = torch.norm(self.quad_demo.b_test - self.b_est_test) / self.b_est_test.shape[0]
        self.w_error_test = torch.norm(self.quad_demo.w_test - self.w_est_test) / self.w_est_test.shape[0]
        print("the test MSE of b: {}".format(self.b_error_test))
        print("the test MSE of w: {}".format(self.w_error_test))
        
    def predict(self, x):
        n = x.shape[0]
        b_est, w_est= torch.zeros([n, self.n_dim]), torch.zeros([n, self.n_dim])
        for i in range(n):
            out = self.net(x[i,:])
            b_est[i,:], w_est[i,:] = out[0:self.n_dim], out[self.n_dim:]
        return b_est, w_est
    
    def save_solution(self):
        self.file_path = "./" + "results//"
        pd.DataFrame(self.b_est.detach().numpy()).to_csv(self.file_path + 'b_est.csv', index=0, header=0)
        pd.DataFrame(self.quad_demo.b.detach().numpy()).to_csv(self.file_path + 'b_reality.csv', index=0, header=0)
        pd.DataFrame(self.w_est.detach().numpy()).to_csv(self.file_path + 'w_est.csv', index=0, header=0)
        pd.DataFrame(self.quad_demo.w.detach().numpy()).to_csv(self.file_path + 'w_reality.csv', index=0, header=0)
        pd.DataFrame(self.b_est.detach().numpy() - self.quad_demo.b.detach().numpy()).to_csv(self.file_path + 'b_error.csv', index=0, header=0)
        pd.DataFrame(self.w_est.detach().numpy() - self.quad_demo.w.detach().numpy()).to_csv(self.file_path + 'w_error.csv', index=0, header=0)
        
    

        