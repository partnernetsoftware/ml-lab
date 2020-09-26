# Concepts

* NLP
* NN () 神经网络
* CNN () 卷积神经网络
* RNN - Recurrent Neural Networks
* GRU (Gated Recurrent Units) 门控循环单元
* LSTM (Long Short Term Memory networks) 长短时记忆网络

## w/ PyTorch

```
# 2019
PyTorch提供2种不同层次的类别（class）用于构建循环网络：
* 多层次类别（Multi-layer classes），包括nn.RNN、nn.GRU和nn.LSTM。这些类别的基类（Object）可用于表示深度双向循环神经网络。
* 单元层类别（Cell-level classes），包括nn.RNNCell、nn.GRUCell和nn.LSTMCell。这些类别的基类仅可用于表示单个单元（如简单RNN、LSTM及GRU的单元），即处理输入数据一个时间步长的单元。
* 当神经网络中不需要太多定制时，多层次类别对单元层类别来说，就像是不错的包装类（wrapper）。
* 构建一个双向RNN非常简单，只需在多层次类别中将双向实参设置为True就可以了。
```

## w/ TensorFlow

```
# 2019
TensorFlow提供tf.nn.rnn_cell模块用于构建标准RNN。 tf.nn.rnn_cell模块中最重要的类别包括：
* 单元层类别（Cell level classes）：用于定义RNN的单个单元，即BasicRNNCell、GRUCell和LSTMCell。
* 多RNN单元类别（MultiRNNCell class）：用于堆栈多个单元，以创建深度RNN。
* 随机失活包装类别（DropoutWrapper class）：用于执行dropout正则化。
```

## w/ Keras

```
# 2019
Keras库提供的循环层包括：
· 简单RNN——全连接RNN，其输出被反馈到输入中
· GRU——门控循环单元层
· LSTM——长短时记忆层
TensorFlow、PyTorch和Keras都具有构建常见RNN架构的内置功能。它们的区别在于接口不同。
Keras的接口非常简单，包含一小串定义明确的参数，能够使上述类别的执行更加简单。作为一个能够在TensorFlow上运行的高级API，Keras使得TensorFlow更加简单。TensorFlow和PyTorch两者的灵活性差不多，但是后者的接口更加简洁明了。
```

## Community 社区

```
# 2019
和PyTorch相比，TensorFlow更加成熟，其社区规模比PyTorch和Keras的社区规模加起来还要大得多，用户基数的增长也比PyTorch和Keras要快。
· 有更大规模的社区，如StackOverFlow上的社区，帮助你解决问题
· 有更多的线上学习资料，如博客、视频、课程等
· 能更快掌握最新的深度学习技术
```

# References

* https://zhuanlan.zhihu.com/p/84674849
* https://www.zhihu.com/question/59274399