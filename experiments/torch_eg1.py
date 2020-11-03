# -*- codiing: utf-8 -*-
# ref https://blog.csdn.net/hetengjiao523/article/details/95469692
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd
import pandas as pd
import csv


df = pd.read_csv(r"./data/point.xlsx", header=None, skiprows=1)

df1 = df.iloc[:, 3:7].values
df2 = df.iloc[:, -1].values
xtrain_features = torch.FloatTensor(df1.astype('float').reshape(-1, 4))
xtrain_labels = torch.FloatTensor(df2)
xtrain = torch.unsqueeze(xtrain_features, dim=1)
ytrain = torch.unsqueeze(xtrain_labels, dim=1)
x, y = autograd.Variable(xtrain), autograd.Variable(ytrain)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


model = Net(n_feature=4, n_hidden=10, n_output=1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

num_epochs = 100000
for epoch in range(num_epochs):
    inputs = x
    target = y
    out = model(inputs)
    loss = loss_fn(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print('Epoch [{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, loss.item()))
