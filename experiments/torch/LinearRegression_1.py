import mytorch
import torch

class main(mytorch):

    def __init__(self):
        # print("LinearRegression.__init__")
        super().__init__()
        w = 1
        # self.w = w
        self.linear = torch.nn.Linear(in_features=w, out_features=w) # in/out dimension
        # self.model = torch.nn.Linear(in_features=w, out_features=w) # in/out dimension

    def forward(self, x):
        return self.linear(x)

        # w = list(x.size())[0] # TODO...
        # print('TODO w=', w)
        # #  linear = self.linear
        # # w = 10
        # linear = torch.nn.Linear(w,w)
        # return linear(x) # y = a*x + b 

if __name__ == '__main__':

    _cuda = torch.cuda.is_available()
    _cuda = False # tmp cuda switch

    mod = main()
    if _cuda: mod = mod.cuda()

    # mod_parameters = mod.parameters()
    # print("mod_parameters=",mod_parameters)
    #mod_parameters = mod.parameters()
    #print("mod_parameters=",list(mod_parameters))

    # if _cuda: x = x.cuda()

    from torch.autograd import Variable
    from matplotlib import pyplot as plt

    x = torch.rand(10,1)
    y = torch.rand(10,1)
    print('x=',x)
    print('y=',y)
    plt.plot(x.numpy(), y.numpy(), 'ro', label='org data')

    # xx = Variable(x, requires_grad=True)
    xx = Variable(x)
    print('xx=',xx)

    for i in range(1000):
    #if True:
        out_y = mod(xx)
        print('out_y=', out_y)
        criterion = torch.nn.MSELoss()
        loss = criterion(out_y, y)
        #print('1.lost.data=',loss.data)

        # backward
        mod_parameters = mod.parameters()
        # print("mod_parameters=",mod_parameters)
        optimizer = torch.optim.SGD(mod_parameters, lr=1e-3)
        #optimizer = torch.optim.SGD(mod.parameters(), lr=1e-3)
        optimizer.zero_grad() # reset grad.
        loss.backward()
        optimizer.step()
        #print('lost.data=',loss.data)
        print('lost.data=',loss.data.item())
        #the_loss = loss.data[0]
        #print(the_loss)

    mod.eval()

    # rst = mod.forward(x)
    # print(x, rst)

    # if _cuda:
    #     plt.plot(x.cpu().numpy(),rst.cpu().detach().numpy(),'ro',label='test')
    # else:
    if True:
        # x_train = Variable(torch.rand(10))
        predict = mod(xx).data.numpy()
        plt.plot(xx.detach().numpy(), predict, label='line')
    plt.show()
