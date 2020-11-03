# ref https://yq.aliyun.com/articles/727936
'''
class Conv1d(_ConvNd):
    """
    in_channels (int): 输入通道数，也就是上图中的d=5
    out_channels (int): 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
    kernel_size (int or tuple): 卷积核的大小，上图3组核的的大小分别为4、5、6
    stride (int or tuple, optional): 卷积步长，每一次卷积计算之间的跨度，默认1
    padding (int or tuple, optional): 输入的每一条边补充0的层数，默认0
    dilation (int or tuple, optional): 卷积核元素之间的间距
    groups (int, optional): 输入通道到输出通道的阻塞连接数
    bias (bool, optional): 是否添加偏置项
    """
'''


import torch
import torch.nn as nn

class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super(TimeSeriesCNN, self).__init__()
        kernel_sizes = [4, 5, 6]
        ts_len = 7 # length of time series
        hid_size = 1
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=5,
                    out_channels=2,
                    kernel_size=kernel_size,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(
            in_features=hid_size * len(kernel_sizes),
            out_features=3,
        )

    def forward(self, x):
        output = [conv(x) for conv in self.convs]
        output = torch.cat(output, dim=1)
        output = output.view(output.size(1), -1)
        output = self.fc(output)
        return output
