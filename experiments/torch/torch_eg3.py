# https://blog.csdn.net/ssjdoudou/article/details/103570011
# 用Pytorch多特征预测股票（LSTM、Bi-LSTM、GRU）


## core

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=30, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


## build network

class GRUNet(nn.Module):
 
    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(128, 1)
        )
 
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        print(out.shape)
        return out
 
 
class LSTMNet(nn.Module):
 
    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )
 
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        print(out.shape)
        return out
 
 
class BiLSTMNet(nn.Module):
 
    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(128, 1)
        )
 
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        print(out.shape)
        return out

## train
net = GRUNet(features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
# start training
for e in range(1000):
    for i, (X, y) in enumerate(train_loader):
        var_x = Variable(X)
        var_y = Variable(y)
        # forward
        out = net(var_x)
        loss = criterion(out, var_y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            torch.save(obj=net.state_dict(), f='models/lstmnetpro_gru_%d.pth' % (e + 1))
 
torch.save(obj=net.state_dict(), f="models/lstmnet_gru_1000.pth")
