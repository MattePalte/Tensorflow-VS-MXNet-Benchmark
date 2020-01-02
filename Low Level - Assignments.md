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

Arguments:
data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".

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
Flattens the input array into a 2-D array by collapsing the higher dimensions.
For an input array with shape (d1, d2, ..., dk), flatten operation reshapes the input array into an output array of shape (d1, d2*...*dk).
Note that the behavior of this function is different from numpy.ndarray.flatten, which behaves similar to mxnet.ndarray.reshape((-1,)).

Inputs -> data: input tensor with arbitrary shape (N, x1, x2, …, xn)

Output -> out: 2D tensor with shape: (N, x1 cdot x2 cdot … cdot xn)

Parameters
data (NDArray) – Input array.
out (NDArray, optional) – The output NDArray to hold the result.
Returns
out – The output of this function.
Return type
NDArray or list of NDArrays

## Normalize a vector:
### MXnet
mxnet.ndarray.norm
mxnet.ndarray.norm(data=None, ord=_Null, axis=_Null, keepdims=_Null, out=None, name=None, **kwargs)
Computes the norm on an NDArray.
This operator computes the norm on an NDArray with the specified axis, depending on the value of the ord parameter. By default, it computes the L2 norm on the entire array. Currently only ord=2 supports sparse ndarrays.

Parameters
data (NDArray) – The input
ord (int, optional, default='2') – Order of the norm. Currently ord=1 and ord=2 is supported.
axis (Shape or None, optional, default=None) –
The axis or axes along which to perform the reduction.
The default, axis=(), will compute over all elements into a scalar array with shape (1,). If axis is int, a reduction is performed on a particular axis. If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed.
keepdims (boolean, optional, default=0) – If this is set to True, the reduced axis is left in the result as dimension with size one.
out (NDArray, optional) – The output NDArray to hold the result.
Returns
out – The output of this function.
Return type
NDArray or list of NDArrays

```Python
x = [[[1, 2],
      [3, 4]],
     [[2, 2],
      [5, 6]]]

norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
                          [5.3851647 6.3245554]]

norm(x, ord=1, axis=1) = [[4., 6.],
                          [7., 8.]]

rsp = x.cast_storage('row_sparse')

norm(rsp) = [5.47722578]

csr = x.cast_storage('csr')

norm(csr) = [5.47722578]
```
### Tensorflow
#### tf.linalg.normalize
Normalizes tensor along dimension axis using specified norm.
tf.linalg.normalize(
    tensor,
    ord='euclidean',
    axis=None,
    name=None
)
This uses tf.linalg.norm to compute the norm along axis.
This function can compute several different vector norms (the 1-norm, the Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).

Args:
tensor: Tensor of types float32, float64, complex64, complex128
ord: Order of the norm. Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number yielding the corresponding p-norm. Default is 'euclidean' which is equivalent to Frobenius norm if tensor is a matrix and equivalent to 2-norm for vectors. Some restrictions apply: a) The Frobenius norm 'fro' is not defined for vectors, b) If axis is a 2-tuple (matrix norm), only 'euclidean', 'fro', 1, 2, np.inf are supported. See the description of axis on how to compute norms for a batch of vectors or matrices stored in a tensor.
axis: If axis is None (the default), the input is considered a vector and a single vector norm is computed over the entire set of values in the tensor, i.e. norm(tensor, ord=ord) is equivalent to norm(reshape(tensor, [-1]), ord=ord). If axis is a Python integer, the input is considered a batch of vectors, and axis determines the axis in tensor over which to compute vector norms. If axis is a 2-tuple of Python integers it is considered a batch of matrices and axis determines the axes in tensor over which to compute a matrix norm. Negative indices are supported. Example: If you are passing a tensor that can be either a matrix or a batch of matrices at runtime, pass axis=[-2,-1] instead of axis=None to make sure that matrix norms are computed.
name: The name of the op.

Returns:
normalized: A normalized Tensor with the same shape as tensor.
norm: The computed norms with the same shape and dtype tensor but the final axis is 1 instead. Same as running tf.cast(tf.linalg.norm(tensor, ord, axis keepdims=True), tensor.dtype).
Raises:
ValueError: If ord or axis is invalid.

## Read a csv dataset:
### MXnet
MXNet provides `CSVIter <http://mxnet.io/api/python/io/io.html#mxnet.io.CSVIter>`__ to read from CSV files and can be used as below:
#lets save `data` into a csv file first and try reading it back
```
np.savetxt('data.csv', data, delimiter=',')
data_iter = mx.io.CSVIter(data_csv='data.csv', data_shape=(3,), batch_size=30)
for batch in data_iter:
    print([batch.data, batch.pad])
```
### Tensorflow
The tf.data API makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations.
The tf.data API introduces a tf.data.Dataset abstraction that represents a sequence of elements, in which each element consists of one or more components. For example, in an image pipeline, an element might be a single training example, with a pair of tensor components representing the image and its label.
There are two distinct ways to create a dataset:
    A data source constructs a Dataset from data stored in memory or in one or more files.
    A data transformation constructs a dataset from one or more tf.data.Dataset objects.

##### Load CSV data
How to load CSV data from a file into a tf.data.Dataset.
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
You can load this using pandas, and pass the NumPy arrays to TensorFlow. If you need to scale up to a large set of files, or need a loader that integrates with TensorFlow and tf.data then use the tf.data.experimental.make_csv_dataset function:
The only column you need to identify explicitly is the one with the value that the model is intended to predict.
LABEL_COLUMN = 'survived'
LABELS = [0, 1]
Now read the CSV data from the file and create a dataset.
(For the full documentation, see tf.data.experimental.make_csv_dataset)
```
def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
```
## Perform the train/test split
### MXnet
##### gluonts.dataset.split.splitter module
Train/test splitter
This module defines strategies to split a whole dataset into train and test subsets.
For uniform datasets, where all time-series start and end at the same point in time OffsetSplitter can be used:
```
splitter = OffsetSplitter(prediction_length=24, split_offset=24)
train, test = splitter.split(whole_dataset)
```
For all other datasets, the more flexible DateSplitter can be used:
```
splitter = DateSplitter(
    prediction_length=24,
    split_date=pd.Timestamp('2018-01-31', freq='D')
)
train, test = splitter.split(whole_dataset)
``` 
The module also supports rolling splits:
```
splitter = DateSplitter(
    prediction_length=24,
    split_date=pd.Timestamp('2018-01-31', freq='D')
)
train, test = splitter.rolling_split(whole_dataset, windows=7)
```
### Tensorflow
##### tfds.Split
Class Split
Enum for dataset splits.
Datasets are typically split into different subsets to be used at various stages of training and evaluation.
TRAIN: the training data.
VALIDATION: the validation data. If present, this is typically used as evaluation data while iterating on a model (e.g. changing hyperparameters, model architecture, etc.).
TEST: the testing data. This is the data to report metrics on. Typically you do not want to use this during model iteration as you may overfit to it.
ALL: Special value, never defined by a dataset, but corresponding to all defined splits of a dataset merged together.
Create a custom split with tfds.Split('custom_name').

