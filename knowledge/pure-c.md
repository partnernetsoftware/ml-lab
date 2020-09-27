
# AI with pure-c

##

https://github.com/100/Cranium

```
# 2016
Cranium is a portable, header-only, feedforward artificial neural network library written in vanilla C99.
It supports fully-connected networks of arbitrary depth and structure, and should be reasonably fast as it uses a matrix-based approach to calculations. It is particularly suitable for low-resource machines or environments in which additional dependencies cannot be installed.
Cranium supports CBLAS integration. Simply uncomment line 7 in matrix.h to enable the BLAS sgemm function for fast matrix multiplication.
```

##

https://github.com/attractivechaos/kann

```
KANN is a standalone and lightweight library in C for constructing and training small to medium artificial neural networks such as multi-layer perceptrons, convolutional neural networks and recurrent neural networks (including LSTM and GRU). It implements graph-based reverse-mode automatic differentiation and allows to build topologically complex neural networks with recurrence, shared weights and multiple inputs/outputs/costs. In comparison to mainstream deep learning frameworks such as TensorFlow, KANN is not as scalable, but it is close in flexibility, has a much smaller code base and only depends on the standard C library. In comparison to other lightweight frameworks such as tiny-dnn, KANN is still smaller, times faster and much more versatile, supporting RNN, VAE and non-standard neural networks that may fail these lightweight frameworks.

KANN could be potentially useful when you want to experiment small to medium neural networks in C/C++, to deploy no-so-large models without worrying about dependency hell, or to learn the internals of deep learning libraries.
```

##

http://leenissen.dk/fann/wp/

```
Fast Artificial Neural Network Library is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks. Cross-platform execution in both fixed and floating point are supported. It includes a framework for easy handling of training data sets. It is easy to use, versatile, well documented, and fast. Bindings to more than 20 programming languages are available. An easy to read introduction article and a reference manual accompanies the library with examples and recommendations on how to use the library. Several graphical user interfaces are also available for the library.
```