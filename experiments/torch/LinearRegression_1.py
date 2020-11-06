import mytorch
import torch

class main(mytorch):

    def __init__(self):
        # print("LinearRegression.__init__")
        super().__init__()
        # self.linear = torch.nn.Linear(in_features=10, out_features=10) # in/out dimension (1=>1)

    def forward(self, x):
        w = list(x.size())[0] # TODO...
        print('w=', w)
        # linear = self.linear
        # w = 10
        linear = torch.nn.Linear(w,w)
        return linear(x) # y = a*x + b 

if __name__ == '__main__':

    _cuda = torch.cuda.is_available()
    _cuda = False # tmp cuda switch

    mod = main()
    if _cuda: mod = mod.cuda()

    # if _cuda: x = x.cuda()

    from torch.autograd import Variable
    x = torch.rand(10)
    print('x=',x)
    xx = Variable(x, requires_grad=True)
    # xx = Variable(x)
    print('xx=',xx)

    y = torch.rand(10)
    print('y=',y)

    out_y = mod(xx)
    print('out_y=', out_y)
    criterion = torch.nn.MSELoss()
    loss = criterion(out_y, y)
    print('lost.data=',loss.data)

    # backward
    mod_parameters = mod.parameters()
    print("mod_parameters=",mod_parameters)
    optimizer = torch.optim.SGD(mod_parameters, lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('lost.data=',loss.data)

    # rst = mod.forward(x)
    # print(x, rst)

    # from matplotlib import pyplot as plt
    # if _cuda:
    #     plt.plot(x.cpu().numpy(),rst.cpu().detach().numpy(),'ro',label='test')
    # else:
    #     plt.plot(x.numpy(),rst.detach().numpy(),'ro',label='test')
    # plt.show()
