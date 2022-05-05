import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np

class Train():
    def __init__(self, net, kkt_condition, BATCH_SIZE):
        self.errors = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.kkt_condition = kkt_condition

    def train(self, epoch, lr):
        optimizer = optim.Adam(self.net.parameters(), lr)
        avg_loss = 0
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.kkt_condition.loss_func(self.BATCH_SIZE)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 100 == 99:
                loss = avg_loss/50
                print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0

                error = self.kkt_condition.loss_func(2**8)
                self.errors.append(error.detach())

    def get_errors(self):
        return self.errors

    def save_model(self):
        torch.save(self.net, 'net_model.pkl')
    
    def plot_kpi(self):
        fig = plt.figure()
        plt.plot(np.log(self.errors), '-b', label='Errors')
        plt.title('Training Loss', fontsize=10)
        plt.show()