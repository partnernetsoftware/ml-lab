# Ref P93

import torch
from torch import nn
from torch.nn import Sequential,Conv2d

class Lenet(nn.Module):

    def __init__(self):

        super(Lenet, self).__init__()

        layer1 = Sequential()
        layer1.add_module('conv1', Conv2d(3,32,3,1, padding=1))
        self.layer1 = layer1 # TODO self.layer_a.append()

    def forward(self, x):
        rt = self.layer1(x)
        rt = rt.view(rt.size(0),-1)
        return rt


if __name__ == '__main__':
    x = torch.rand(4)
    y = torch.rand(4)
    print('x=', x,)
    print('y=', y)
    print('x*y=', x*y)
    print('y*x=', y*x)
    a = torch.ones(3,4)
    print('a=')
    # net = Lenet()
    # net.forward()
