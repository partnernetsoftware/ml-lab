import torch
import time

VERSION = '20201111'

class constructor(torch.nn.Module):

    def __init__(self):
        #self.super_time = time.time() # for performance profiling
        super(constructor, self).__init__()
        self.init_time = time.time() # for performance profiling

    def dump(self):
        print('{')
        print('"VERSION":',VERSION)
        print(',"__name__":"{}"'.format(__name__))
        print(',"init_time":"{}"'.format(self.init_time))
        print('}')

if __name__ == '__main__':
    import mytorch
    mytorch().dump()
else: # make this module callable, i.e. import mytorch; mytorch.ver()
    import sys
    sys.modules[__name__] = constructor
