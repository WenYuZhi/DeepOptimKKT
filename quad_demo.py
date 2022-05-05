import numpy as np
import pandas as pd
import torch
import random
from pyomo.environ import *
from tqdm import tqdm
from sklearn import preprocessing

class Quad_Demo:
    def __init__(self, n_dim, d_input, d_output, n_constrs = 0) -> None:
        self.n_dim, self.n_constrs = n_dim, n_constrs
        self.d_input, self.d_output = d_input, d_output
        assert(self.d_output == self.n_dim)
        self.A = torch.randn([self.n_dim, self.n_dim])
        self.A = torch.addmm(beta = 1, input = torch.eye(self.n_dim), mat1=torch.t(self.A), mat2=self.A)
      
    def get_uncertainty(self, n_sample, x_up, x_down):
        self.n_sample, self.x_up, self.x_down = n_sample, x_up, x_down
        assert(x_down < x_up)
        self.x = self.random_sample(size = self.n_sample)
        self.b = self.predict(self.x)
    
    def transforms(self):
        self.scaler = preprocessing.StandardScaler().fit(self.x)
        self.x = torch.Tensor(self.scaler.transform(self.x))

    def random_sample(self, size = 2**10):
        x = (self.x_up-self.x_down)*torch.rand([size, self.d_input]) + self.x_down*torch.ones([size, self.d_input])
        return x
    
    def sample(self, size = 2**10):
        assert(size <= self.n_train)
        ind = random.sample([i for i in range(self.n_train)], size)
        x, b, w = self.x[ind,:], self.b[ind,:], self.w[ind,:]
        return x,b,w

    def predict(self, x):
        self.F = torch.rand([self.d_input, self.d_output])
        # self.b = torch.mm(x, self.F)**2
        self.b = torch.sin(torch.mm(x, self.F))
        # self.b = torch.mm(x, self.F)
        return self.b
    
    def bulid_model(self):
        self.model = ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.n_dim)])
        self.model.w = Var(self.model.I, within=Reals)
        self.model.A = Param(self.model.I, self.model.I, initialize=self.a_data)
        self.model.b = Param(self.model.I, default = 0.0, within = Reals, mutable=True)
        self.set_objective()
    
    def a_data(self, model, i, j):
        return float(self.A[i,j])
    
    def set_objective(self):
        self.model.obj = Objective(rule=self.obj_rule, sense=minimize)
    
    def obj_rule(self, model):
        expr = summation(self.model.b, self.model.w)
        for i in self.model.I:
            for j in self.model.I:
                expr += self.model.A[i,j]*self.model.w[i]*self.model.w[j]
        return expr
    
    def solve(self):
        self.obj_values, self.w = [], []
        for i in range(self.n_sample):
            self.set_params(self.b[i,:])
            opt = SolverFactory('gurobi')
            solution = opt.solve(self.model)
            self.obj_values.append(value(self.model.obj))
            self.w.append([value(self.model.w[j]) for j in self.model.I])
        self.obj_values, self.w = torch.Tensor(self.obj_values), torch.Tensor(self.w)
    
    def set_params(self, b):
        for i in range(len(self.model.b)):
            self.model.b[i] = float(b[i]) 
    
    def write_model(self):
        self.model.write("qp_model.lp")
    
    def load_data(self, file_path, n_sample, x_up, x_down):
        self.n_sample, self.x_up, self.x_down = n_sample, x_up,  x_down
        self.A = pd.read_csv(file_path + 'A.csv', names = [i for i in range(self.n_dim)])
        self.b = pd.read_csv(file_path + 'b.csv', names = [i for i in range(self.d_output)])
        self.F = pd.read_csv(file_path + 'F.csv', names = [i for i in range(self.d_output)])
        self.w = pd.read_csv(file_path + 'w.csv', names = [i for i in range(self.n_dim)])
        self.x = pd.read_csv(file_path + 'x.csv', names = [i for i in range(self.d_input)])
        self.obj_values = pd.read_csv(file_path + 'obj_values.csv', names=['objvalues'])
        self.__assert_data()
        self.__to_tensor()
    
    def __assert_data(self):
        assert(self.A.shape[0] == self.n_dim and self.A.shape[1] == self.n_dim)
        assert(self.n_sample == self.b.shape[0])
        assert(self.n_sample == self.w.shape[0])
        assert(self.n_sample == self.x.shape[0])
        assert(self.n_sample == self.obj_values.shape[0])
    
    def __to_tensor(self):
        self.A, self.b = torch.Tensor(self.A.values), torch.Tensor(self.b.values)
        self.F, self.w = torch.Tensor(self.F.values), torch.Tensor(self.w.values)
        self.x, self.obj_values = torch.Tensor(self.x.values), torch.Tensor(self.obj_values.values)
    
    def validate_stationary(self):
        self.stationarity_error = torch.sum((torch.addmm(input=self.b, mat1=2*self.w, mat2=self.A))**2)
        print("stationarity error: {}".format(self.stationarity_error))
    
    def split_train_test(self, rate = 0.1):
        assert(rate >= 0 and rate <= 0.5)
        self.n_test = int(self.n_sample * rate)
        self.n_train = self.n_sample - self.n_test
        ind = random.sample([i for i in range(self.n_sample)], self.n_test)
        self.x_test, self.b_test, self.w_test = self.x[ind,:], self.b[ind,:], self.w[ind,:]
        ind_train = [i for i in range(self.n_sample) if i not in ind]
        self.x, self.b, self.w = self.x[ind_train,:], self.b[ind_train,:], self.w[ind_train,:]
   
    def save_data(self):
        self.file_path = "./trainning_data//"
        pd.DataFrame(self.A.detach().numpy()).to_csv(self.file_path + 'A.csv', index=0, header=0)
        pd.DataFrame(self.x.detach().numpy()).to_csv(self.file_path + 'x.csv', index=0, header=0)
        pd.DataFrame(self.b.detach().numpy()).to_csv(self.file_path + 'b.csv', index=0, header=0)
        pd.DataFrame(self.w.detach().numpy()).to_csv(self.file_path + 'w.csv', index=0, header=0)
        pd.DataFrame(self.obj_values.detach().numpy()).to_csv(self.file_path + 'obj_values.csv', index=0, header=0)
        pd.DataFrame(self.F.detach().numpy()).to_csv(self.file_path + 'F.csv', index=0, header=0)


        


