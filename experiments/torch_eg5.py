# https://blog.csdn.net/rizero/article/details/104244454
# 股票成交量预测（Pytorch基础练习）

## depends

import pandas as pd

import torch
import torch.nn
import torch.optim

from debug import ptf_tensor

## raw data

url = 'C:/Users/HUAWEI/Desktop/深度学习/Blog附带代码/FB.csv'
df = pd.read_csv(url, index_col=0) #读取全部数据
index_col = ['col_1','col_2']   # 读取指定的几列
error_bad_lines = False   # 当某行数据有问题时，不报错，直接跳过，处理脏数据时使用
na_values = 'NULL'   # 将NULL识别为空值

## data clean

#数据集的处理
'''
因为数据是日期新的占index靠前
'''
train_start, train_end=sum(df.index>='2017'),sum(df.index>='2013') 
test_start, test_end=sum(df.index>='2018'),sum(df.index>='2017')

n_total_train = train_end -train_start
n_total_test = test_end -test_start

s_mean=df[train_start:train_end].mean() #计算均值，为归一化做准备
s_std=df[train_start:train_end].std() # 计算标准差，为归一化做准备

n_features=5 # 五个特征量

#选取col from 0-4 也就是Open，High，Low，Close，Volume，并进行归一化
df_feature=((df-s_mean)/s_std).iloc[:,:n_features] 

s_labels=(df['Volume']<df['Volume'].shift(1)).astype(int)
##.shift(1)把数据下移一位
#用法参见：https://www.zhihu.com/question/264963268

#label建立的标准：假如今天次日的成交量大于当日的成交量，标签=1，反之=0


## alter format

x=torch.tensor(df_feature.values,dtype=torch.float32) # size: [m,5]
ptf_tensor(x,'x')
y=torch.tensor(s_labels.values.reshape(-1,1),dtype=torch.float32) # size [m,1]
ptf_tensor(y,'y')

##  build nn

fc=torch.nn.Linear(n_features,1)
weights,bias=fc.parameters()
criterion=torch.nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(fc.parameters())

## train w+ check

n_steps=20001 #迭代20001次

for step in range(n_steps):
    if step:
        optimizer.zero_grad() # 梯度清零，不然会叠加的
        loss.backward() # 计算参数的梯度
        optimizer.step() # 根据参数梯度结果迭代推出新的参数
    
    pred=fc(x) # 计算预测结果
    loss=criterion(pred[train_start:train_end],y[train_start:train_end]) #计算预测的损失

    if step % 500==0:
        #print('#{}, 损失 = {:g}'.format(step, loss))        
        output = (pred > 0)
        correct = (output == y.bool())
        n_correct_train = correct[train_start:train_end].sum().item() #计算训练正确的数量
        n_correct_test = correct[test_start:test_end].sum().item() #计算测试正确的数量
        accuracy_train = n_correct_train / n_total_train #计算精确度
        accuracy_test = n_correct_test / n_total_test
        print('训练集准确率 = {}, 测试集准确率 = {}'.format(accuracy_train, accuracy_test))
        
## 



