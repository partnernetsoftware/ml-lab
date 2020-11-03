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
    
class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.use_element = config.use_element
        self.config = config
 
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=config.embedding_size, 
                                        out_channels=config.feature_size, 
                                        kernel_size=h),
#                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=config.max_text_len-h+1))
                     for h in config.window_sizes
                    ])
        self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),
                            out_features=config.num_class)
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))    
    
    def forward(self, x):
        embed_x = self.embedding(x)
        
        #print('embed size 1',embed_x.size())  # 32*35*256
# batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
        #print('embed size 2',embed_x.size())  # 32*256*35
        out = [conv(embed_x) for conv in self.convs]  #out[i]:batch_size x feature_size*1
        #for o in out:
        #    print('o',o.size())  # 32*100*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        #print(out.size(1)) # 32*400*1
        out = out.view(-1, out.size(1)) 
        #print(out.size())  # 32*400 
        if not self.use_element:
            out = F.dropout(input=out, p=self.dropout_rate)
            out = self.fc(out)
        return out
    
    conv1 = nn.Conv1d(in_channels=256，out_channels=100,kernel_size=2)
    input = torch.randn(32,35,256)
    # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    input = input.permute(0,2,1)
    out = conv1(input)
    print(out.size())
