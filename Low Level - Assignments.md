# Assignments

1. Select the 10-15 most important fundamental (low-level) operations in the NN architecture

    - define a variable
    - activation function
    - create a vector
    - dot product
    - matrix multiplication
    - flatten a vector
    - normalize a vector
    - define a kernel
    - for loop for a kernel (convolutional op)
    - read a csv dataset 
    - perform the train/test split 
    
1. Look for those most important operations in both the two API and collect the results in a table

## Fast Math Computation - API
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

# How to define a variable?

### TF: tf.Variable
A TensorFlow variable is the best way to represent shared, persistent state manipulated by your program.

Variables are manipulated via the tf.Variable class. A tf.Variable represents a tensor whose value can be changed by running ops on it. Specific ops allow you to read and modify the values of this tensor. Higher level libraries like tf.keras use tf.Variable to store model parameters. 

This guide covers how to create, update, and manage tf.Variables in TensorFlow. https://www.tensorflow.org/guide/variable

### MX: mxnet.symbol
The Symbol API in Apache MXNet is an interface for symbolic programming. It features the use of computational graphs, reduced memory usage, and pre-use function optimization.

Guide (not in python unluckily): https://mxnet.apache.org/api/perl/docs/tutorials/symbol 

# Which activation functions are present?

### TF: tf.keras.activations.relu (LOW) - tf.nn.relu (HIGH)

https://www.tensorflow.org/api_docs/python/tf/keras/activations

Functions in tf.keras.activations:
- elu(...): Exponential linear unit.
- exponential(...): Exponential activation function.
- hard_sigmoid(...): Hard sigmoid activation function.
- linear(...): Linear activation function.
- relu(...): Rectified Linear Unit.
- selu(...): Scaled Exponential Linear Unit (SELU).
- sigmoid(...): Sigmoid.
- softmax(...): The softmax activation function transforms the - outputs so that all values are in
- softplus(...): Softplus activation function.
- softsign(...): Softsign activation function.
- tanh(...): Hyperbolic Tangent (tanh) activation function.
```python
tf.keras.activations.relu(
    x,
    alpha=0.0,
    max_value=None,
    threshold=0
)
```
Arguments:
- x: A tensor or variable.
- alpha: A scalar, slope of negative section (default=0.).
- max_value: float. Saturation threshold.
- threshold: float. Threshold value for thresholded activation.

Returns:
A tensor.

### MX: mxnet.ndarray.Activation
https://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.Activation.html#mxnet.ndarray.Activation

mxnet.ndarray.Activation(data=None, act_type=_Null, out=None, name=None, **kwargs)
Applies an activation function element-wise to the input.

Arguments:
- data (NDArray) – The input array.
- act_type ({'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required) – Activation function to be applied.
- out (NDArray, optional) – The output NDArray to hold the result.

Return type: NDArray or list of NDArrays

# How to create vector and dot product?

### TF: tf.keras.backend.dot
Alternatives: tf.tensordot

Create a tensor: https://www.tensorflow.org/api_docs/python/tf/Tensor
```python
# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)
```

Multiplies 2 tensors (and/or variables) and returns a tensor.
```python
tf.keras.backend.dot(x, y)
```
Arguments:
- x: Tensor or variable.
- y: Tensor or variable.

Returns: A tensor, dot product of x and y.

### MX: mxnet.nd -> mxnet.ndarray.dot¶
Guide: https://gluon.mxnet.io/chapter01_crashcourse/ndarray.html
Dot-product: https://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.dot.html

More or less like Numpy
```python
x = nd.ones(shape=(3,3))
print('x = ', x)
y = nd.arange(3)
print('y = ', y)
```
y =
[ 0.  1.  2.]
<NDArray 3 @cpu(0)>

x =<br>
[[ 1.  1.  1.] <br>
 [ 1.  1.  1.]<br>
 [ 1.  1.  1.]]<br>
<NDArray 3x3 @cpu(0)>

#### Differences with Numpy
mxnet.ndarray is similar to numpy.ndarray in some aspects. But the differences are not negligible. For instance:

- mxnet.ndarray.NDArray.T does real data transpose to return new a copied array, instead of returning a view of the input array.

- mxnet.ndarray.dot performs dot product <span style="color:red">between the last axis of the first input array and the first axis of the second input</span>, while <span style="color:red">numpy.dot uses the second last axis of the input array.</span>

#### mxnet.ndarray.dot
Dot product of two arrays.
dot’s behavior depends on the input array dimensions:
- 1-D arrays: inner product of vectors
- 2-D arrays: matrix multiplication
- N-D arrays: a sum product over the last axis of the first input and the first axis of the second input

For example, given 3-D x with shape (n,m,k) and y with shape (k,r,s), the result array will have shape (n,m,r,s). It is computed by:

## Matrix Multiplication

### TF: tf.linalg.matmul

Multiplies matrix a by matrix b, producing a * b.
Note: This is matrix product, not element-wise product (refer to tf.math.multiply for that).

Parameters:
- a: Tensor of type float16, float32, float64, int32, complex64, complex128 and rank > 1.
- b: Tensor with same type and rank as a.
- transpose_a: If True, a is transposed before multiplication.
- transpose_b: If True, b is transposed before multiplication.
- adjoint_a: If True, a is conjugated and transposed before multiplication.
- adjoint_b: If True, b is conjugated and transposed before multiplication.
- a_is_sparse: If True, a is treated as a sparse matrix.
- b_is_sparse: If True, b is treated as a sparse matrix.
- name: Name for the operation (optional).

Returns:

- A Tensor of the same type as a and b where each inner-most matrix is the product of the corresponding matrices in a and b, e.g. if all transpose or adjoint attributes are False: 
output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.


```python
# 2-D tensor `a`
# [[1, 2, 3],
#  [4, 5, 6]]
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

# 2-D tensor `b`
# [[ 7,  8],
#  [ 9, 10],
#  [11, 12]]
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])

# `a` * `b`
# [[ 58,  64],
#  [139, 154]]
c = tf.matmul(a, b)

# Since python >= 3.5 the @ operator is supported (see PEP 465).
# In TensorFlow, it simply calls the `tf.matmul()` function, so the
# following lines are equivalent:
d = a @ b @ [[10.], [11.]]
d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
```

### MX: mxnet.ndarray.linalg.gemm2

mxnet.ndarray.linalg.gemm2(A=None, B=None, transpose_a=_Null, transpose_b=_Null, alpha=_Null, axis=_Null, out=None, name=None, **kwargs)

Parameters:
- A (NDArray) – Tensor of input matrices
- B (NDArray) – Tensor of input matrices
- transpose_a (boolean, optional, default=0) – Multiply with transposed of first input (A).
- transpose_b (boolean, optional, default=0) – Multiply with transposed of second input (B).
- alpha (double, optional, default=1) – Scalar factor multiplied with A*B.
- axis (int, optional, default='-2') – Axis corresponding to the matrix row indices.
- out (NDArray, optional) – The output NDArray to hold the result.

Returns:
- out – The output of this function. Type: NDArray or list of NDArrays

Performs general matrix multiplication. Input are tensors A, B, each of dimension n >= 2 and having the same shape on the leading n-2 dimensions.
If n=2, the BLAS3 function gemm is performed:

out = alpha * op(A) * op(B)

Here alpha is a scalar parameter and op() is either the identity or the matrix transposition (depending on transpose_a, transpose_b).

If n>2, gemm is performed separately for a batch of matrices. The column indices of the matrices are given by the last dimensions of the tensors, the row indices by the axis specified with the axis parameter. By default, the trailing two dimensions will be used for matrix encoding.
```python
Single matrix multiply
A = [[1.0, 1.0], [1.0, 1.0]]
B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
gemm2(A, B, transpose_b=True, alpha=2.0)
         = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]

Batch matrix multiply
A = [[[1.0, 1.0]], [[0.1, 0.1]]]
B = [[[1.0, 1.0]], [[0.1, 0.1]]]
gemm2(A, B, transpose_b=True, alpha=2.0)
        = [[[4.0]], [[0.04 ]]]
```

## How to flatten a layer

### TF: tf.keras.layers.Flatten
Flattens the input. Does not affect the batch size. If inputs are shaped (batch,) without a channel dimension, then flattening adds an extra channel dimension and output shapes are (batch, 1).
```python
model = Sequential()
model.add(Convolution2D(filters = 64, kernel_size = 3, strides = 3,
                        border_mode='same',
                        input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536) 
```

### MX: mxnet.gluon.nn.Flatten
Flattens the input to two dimensional.

Inputs -> data: input tensor with arbitrary shape (N, x1, x2, …, xn)

Output -> out: 2D tensor with shape: (N, x1 cdot x2 cdot … cdot xn)