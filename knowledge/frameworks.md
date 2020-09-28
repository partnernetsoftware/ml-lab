# Comparison

https://en.wikipedia.org/wiki/Comparison_of_deep-learning_software


# TensorFlow

* Official web:
* Leading Authors: Google Brain
* License: Apache 2.0
* Lang: C++/Python

```
Google Brain基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理
于2015年11月9日发布，
于2017年12月份预发布动态图机制Eager Execution。 

TensorFlow主要支持静态计算图的形式，计算图的结构比较直观，但是在调试过程中十分复杂与麻烦，一些错误更加难以发。但是2017年底发布了动态图机制Eager Execution，加入对于动态计算图的支持，但是目前依旧采用原有的静态计算图形式为主。TensorFlow拥有TensorBoard应用，可以监控运行过程，可视化计算图。 

```

* https://github.com/tensorflow/tfjs-core
* https://github.com/tensorflow/tfjs
* https://github.com/chaosmail/deeplearnjs-caffe

# Keras

```
Keras是一个用Python编写的开源神经网络库，它能够在TensorFlow，CNTK，Theano或MXNet上运行。旨在实现深度神经网络的快速实验，它专注于用户友好，模块化和可扩展性。其主要作者和维护者是Google工程师FrançoisChollet。
Keras是基于多个不同框架的高级API，可以快速的进行模型的设计和建立，同时支持序贯和函数式两种设计模型方式，可以快速的将想法变为结果，但是由于高度封装的原因，对于已有模型的修改可能不是那么灵活。 
```

# Apache MXNet

* Leading Authors: Intel, Baidu, Microsoft, Wolfram Research, etc
* Lang: Python/C++/R/...

```
MXNet是DMLC（Distributed Machine Learning Community）开发的一款开源的、轻量级、可移植的、灵活的深度学习库，它让用户可以混合使用符号编程模式和指令式编程模式来最大化效率和灵活性，目前已经是AWS官方推荐的深度学习框架。MXNet的很多作者都是中国人，其最大的贡献组织为百度。
MXNet同时支持命令式和声明式两种编程方式，即同时支持静态计算图和动态计算图，并且具有封装好的训练函数，集灵活与效率于一体，同时已经推出了类似Keras的以MXNet为后端的高级接口Gluon。 
```

# PyTorch

* Lang: Python
* not yet support mobile?

```
PyTorch是Facebook于2017年1月18日发布的python端的开源的深度学习库，基于Torch。支持动态计算图，提供很好的灵活性。2018年5月份的开发者大会上，Facebook宣布实现PyTorch与Caffe2无缝结合的PyTorch1.0版本将马上到来。 
PyTorch为动态计算图的典型代表，便于调试，并且高度模块化，搭建模型十分方便，同时具备及其优秀的GPU支持，数据参数在CPU与GPU之间迁移十分灵活
```

# Caffe

* official web: https://caffe.berkeleyvision.org/
* author: http://daggerfs.com/
* repo: https://github.com/BVLC/caffe ( by BVLC - Berkeley Vision and Learning Center )

```
A deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research (BAIR) and by community contributors.
Yangqing Jia created the project during his PhD at UC Berkeley.
Caffe is released under the BSD 2-Clause license.
```

Quick tiny example (TODO)

```
# @ref https://zhuanlan.zhihu.com/p/24087905
caffe.set_mode_cpu() # CPU
caffe.set_device(0) # GPU
caffe.set_mode_gpu()
net = caffe.Net('conv.prototxt', caffe.TEST) #load network proto
net.blobs #for input data and its propagation in the layers 
net.params #a vector of blobs for weight and bias parameters
net.forward() # forward broadcasting
net.save('mymodel.caffemodel')
```

## google scholar citations

https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=-ltRSM0AAAAJ:u5HHmVD_uO8C

# misc

CNTK, Theano, DeepLearning4, Lasagne, Neon, etc

# Learning Curves

```
# 2018
对于框架本身的语言设计来讲，TensorFlow是比较不友好的，与Python等语言差距很大，有点像基于一种语言重新定义了一种编程语言，并且在调试的时候比较复杂。每次版本的更新，TensorFlow的各种接口经常会有很大幅度的改变，这也大大增加了对其的学习时间；Keras是一种高级API，基于多种深度学习框架，追求简洁，快速搭建模型，具有完美的训练预测模块，简单上手，并能快速地将所想变现，十分适合入门或者快速实现。但是学习会很快遇到瓶颈，过度的封装导致对于深度学习知识的学习不足以及对于已有神经网络层的改写十分复杂；MXNet同时支持命令式编程和声明式编程，进行了无缝结合，十分灵活，具备完整的训练模块，简单便捷，同时支持多种语言，可以减去学习一门新主语言的时间。上层接口Gluon也极其容易上手；PyTorch支持动态计算图，追求尽量少的封装，代码简洁易读，应用十分灵活，接口沿用Torch，具有很强的易用性，同时可以很好的利用主语言Python的各种优势。对于文档的详细程度，TensorFlow具备十分详尽的官方文档，查找起来十分方便，同时保持很快的更新速度，但是条理不是很清晰，教程众多；Keras由于是对于不同框架的高度封装，官方文档十分详尽，通俗易懂；MXNet发行以来，高速发展，官方文档较为简单，不是十分详细，存在让人十分迷惑的部分，框架也存在一定的不稳定性；PyTorch基于Torch并由Facebook强力支持，具备十分详细条理清晰的官方文档和官方教程。对于社区，庞大的社区可以推动技术的发展并且便利问题的解决，由Google开发并维护的TensorFlow具有最大社区，应用人员团体庞大；Keras由于将问题实现起来简单，吸引了大量研究人员的使用，具有很大的用户社区；MXNet由Amazon，Baidu等巨头支持，以其完美的内存、显存优化吸引了大批用户，DMLC继续进行开发和维护；PyTorch由Facebook支持，并且即将与Caffe2无缝连接，以其灵活、简洁、易用的特点在发布紧一年多的时间内吸引了大量开发者和研究人员，火爆程度依旧在不断攀升，社区也在不断壮大。
# 2020
http://jiagoushi.pro/technology-selectiondifference-between-keras-vs-tensorflow-vs-pytorch
数据科学家在深度学习中选择的最顶尖的三个开源库框架是PyTorch、TensorFlow和Keras。Keras是一个用python脚本编写的神经网络库，可以在TensorFlow的顶层执行。它是专门为深度神经网络的鲁棒执行而设计的。TensorFlow是一种在数据流编程和机器学习应用中用于执行多个任务的工具。PyTorch是一个用于自然语言处理的机器学习库。

Caffe2和PyTorch
PyTorch用来做非常动态变化的研究加上对速度要求不高的产品。
Caffe2用来做计算机视觉，HPC和数值优化的研究，加上产品线里的高效部署。Caffe可以继续用，
不过如果你关注mix precision或者heterogeneous computation或者手机和嵌入式端的话，建议尝试一下Caffe2。
如果Theano，建议转向TensorFlow，或者PyTorch，后者更灵活一些。
如果你用Torch，强烈建议转向PyTorch。已有模型可以考虑torch2caffe来部署。
如果你用TensorFlow，开心就好，performance的确是个问题，但是毕竟社区好。
如果你想认真学machine learning，那请不要用keras，一般收到的反馈是，keras做简单的东西容易，一旦你要做点真research，就很难改，因为包装封装太多。
```

# Performance

```
# 2018
Keras为基于其他深度学习框架的高级API，进行高度封装，计算速度最慢且对于资源的利用率最差；在模型复杂，数据集大，参数数量大的情况下，MXNet和PyTorch对于GPU上的计算速度和资源利用的优化十分出色，并且在速度方面MXNet优化处理更加优秀；相比之下，TensorFlow略有逊色，但是对于CPU上的计算加速，TensorFlow表现更加良好。
```

# Google Trends

* https://trends.google.com/trends/explore?date=today%205-y&geo=US&q=pytorch,%2Fg%2F11bwp1s2k3,%2Fg%2F11c1r2rvnp

# References and Links

* https://www.zhihu.com/question/59274399
* https://github.com/tangyudi/Ai-Learn

