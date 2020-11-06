import mytorch
import torch

class main(mytorch):

    def __init__(self):
        # print("LinearRegression.__init__")
        super().__init__()
        self.linear = torch.nn.Linear(in_features=1,out_features=1) # in/out dimension (1=>1)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    _cuda = torch.cuda.is_available
    mod = main()
    if _cuda: mod = mod.cuda()
    mod = mod.cuda()
    x = torch.rand(1)
    if _cuda: x = x.cuda()
    rst = mod.forward(x)
    print(x, rst)
