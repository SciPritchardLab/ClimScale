#%%
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys

def gen_data():
    torch.manual_seed(1)
    n = 25
    x = (torch.rand((n, 1))-0.5)*3
    x = torch.sort(x.squeeze())[0].unsqueeze(-1)
    ind = ((torch.arange(0, n)+1)/n)*2+0.1
    true_mu = x[:,0] - 2*x[:, 0]**2 + 0.5*x[:,0]**3
    y = true_mu + torch.randn(n)*ind.pow(2)
    y = y.unsqueeze(-1)

    x.requires_grad=True
    
    # test set
    torch.manual_seed(3)
    n_t = 25
    x_t = (torch.rand((n_t, 1))-0.5)*3
    x_t = torch.sort(x_t.squeeze())[0].unsqueeze(-1)
    ind = ((torch.arange(0, n_t)+1)/n_t)*2+0.1
    true_mu_t = x_t[:,0] - 2*x_t[:, 0]**2 + 0.5*x_t[:,0]**3
    y_t = true_mu_t + torch.randn(n_t)*ind.pow(2)
    y_t = y_t.unsqueeze(-1)

    return(x, y, x_t, y_t)

def deeper_net(n_feature=1, n_layers=3, n_hidden=10, n_output=1):
        
        layers = []

        layers.append(torch.nn.Linear(n_feature, n_hidden))
        layers.append(torch.nn.Sigmoid())

        for _ in range(n_layers-2):
            layers.append(torch.nn.Linear(n_hidden, n_hidden))
            layers.append(torch.nn.Sigmoid())
        
        layers.append(torch.nn.Linear(n_hidden, n_output))

        return(nn.Sequential(*layers))

loss_func = torch.nn.MSELoss()
net = deeper_net()
x, y, x_t, y_t = gen_data()

plot_x = torch.unsqueeze(torch.linspace(-1.5, 1.5, 100), dim=1)

opt = torch.optim.Adam(net.parameters(), lr=0.001)
# %%

epochs = 10000
losses = []
writer = SummaryWriter()

for e in range(epochs):
    opt.zero_grad()
    loss = loss_func(net(x), y)
    loss.backward()
    opt.step()

    losses.append(loss)

    if e % 100 == 0:
        
        writer.add_scalar("loss", loss, e)


        fig, ax = plt.subplots()
        ax.scatter(x.detach(), y.detach(), c='g')
        ax.plot(plot_x.detach(), net(plot_x).detach())
        ax.title.set_text("fit over data")


        writer.add_figure('fit', fig, e)
        writer.flush()
# %%
