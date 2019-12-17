# Assignments

## 8 Dec 2019

1. Select the 10-15 most important fundamental (low-level) operations in the NN architecture

    - define a variable
    - activation function
    - dot product
    - matrix multiplication
    - create a vector
    - flatten a vector
    - normalize a vector
    - define a kernel
    - for loop for a kernel (convolutional op)
    - read a csv dataset 
    - perform the train/test split 
    
1. Look for those most important operations in both the two API and collect the results in a table

Each framework has its own low level library for fast math computation:

### MXNet: mxnet.ndarray
The NDArray library in Apache MXNet defines the core data structure for all mathematical computations. NDArray supports fast execution on a wide range of hardware configurations and automatically parallelizes multiple operations across the available hardware.

### TF 2.0: tf.function
Creates a callable TensorFlow graph from a Python function.

It is best to decorate functions that represent the largest computing bottlenecks:
- training loops 
- model’s forward pass).

NB: when you decorate a function with tf.function, you loose some of the benefits of eager execution:
- not be able to setup breakpoints 
- no print() inside that section of code.



