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

1. ### MXNet plus Horovod -> Distributed Deep Learning
    https://medium.com/apache-mxnet/distributed-training-using-apache-mxnet-with-horovod-44f98bf0e7b7

1. ### Everything you need to know about TF 2.0
    Quick and easy overview of TF 2.0 without references to TF 1.
    https://gilberttanner.com/blog/tensorflow-2-0-everything-you-need-to-know

1. ### Various paradigm Declarative/Symbolic/Functional/Imperative in TF 2.0
    https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021

1. ### Paradigm in MXNet
    https://mxnet.apache.org/api/architecture/program_model
    
1. ### Which weights are learned in CNN?
    https://datascience.stackexchange.com/questions/25754/updating-the-weights-of-the-filters-in-a-cnn
    Convolutional layers are different in that they have a fixed number of weights governed by the choice of filter size and number of filters, but independent of the input size.
    Each filter has a separate weight in each position of its shape. So if you use two 3x3x3 filters then you will have 54 weights, again not counting bias. 
    Follow the link for a visualization.

1. ### Tell the differences between the frameworks (tf, gluon ,chainer, pytorch)
https://twitter.com/fchollet/status/1052228463300493312?lang=en


## Official Documentations

1. ### LeNet in Gluon - "Dive into Deep Learning" Guide
    https://d2l.ai/chapter_convolutional-neural-networks/lenet.html

1. ### What are functions in TF 2.0 - How they are related to low level API
    https://www.tensorflow.org/guide/function 

1. ### How to define a network in TF 2.0 - Imperative vs Symbolic
https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html

## Problems

1. ### No Low-Level guide in TF 2.0
    https://github.com/tensorflow/tensorflow/issues/33823
    