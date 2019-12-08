# Tensorflow-VS-MXNet-Benchmark
Comparison of Tensorflow and MXNet framworks using state-of-the-art networks. Some criteria such as training and inference cost, GPU and CPU performance, and software features (API and ...).

# Goal
Abstraction |Tensorflow | MXNet
-------------|--------------|--------------
High Level | e.g. Keras | e.g. Gluon
Low Level | variable API | MXNet Library

### Things to pay attention to:
1. Use same random seed (if possible)
1. Fix the same initialization (in terms of weight)
1. Same hyperparameters (e.g. optimizer, nr epochs, ...)
1. The accuracy should be the same (since the operations must be the same)
1. If not possible to enforce same random seed, run the train multiple times and take the average

### Caracteristics to compare:
The focus is on the train part only.
1. CPU (train time)
1. GPU - if available (train time)
1. Memory usage 
1. nr lines of code
1. readability


# Resources

## Blog Articles

1. ### Tensorflow vs Mxnet - Part 1 - Multi-Layer Perceptron (Dataset = MNIST)
    https://medium.com/@mouryarishik/tensorflow-2-0-vs-mxnet-41edd3b7574f

    Compare nets with timing: 
    * Keras(TF), 
    * Gluon(MXNet), 
    * mxnet-module(MXNet)

    Naive approach in comparing results (e.g. no multiple runs, no enforcement of same initialization, no multiple run average)

1. ### Tensorflow vs Mxnet - Part 2 - LeNet (Dataset = MNIST)
    https://medium.com/@mouryarishik/tensorflow-vs-mxnet-part-2-b14ff20377

    Same author as above, same procedure.

1. ### LeNet in Tensorflow - but 1.4 version :(
    https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086
    The keras version is usable but the low level has to be readapted

## Official Documentations

1. ### LeNet in Gluon - "Dive into Deep Learning" Guide
    https://d2l.ai/chapter_convolutional-neural-networks/lenet.html

