## Major characteristic


### Eager Execution By Default
source: https://hackernoon.com/everything-you-need-to-know-about-tensorflow-2-0-b0856960c074 

TF 1.x : As you might recall, to build a Neural Net in TF 1.x, we needed to define this abstract data structure called a Graph. Also, (as you probably have tried), if we attempted to print one of the graph nodes, we would not see the values we were expecting. Instead, we would see a reference to the graph node. To actually, run the graph, we needed to use an encapsulation called a Session. And using the Session.run() method, we could pass Python data to the graph and actually train our models.

TF 2.0 : With eager execution, this changes. Now, TensorFlow code can be run like normal Python code. Eagerly. Meaning that operations are created and evaluated at once. TensorFlow 2.0 code looks a lot like NumPy code. In fact, TensorFlow and NumPy objects can easily be switched from one to the other. Hence, you do not need to worry about placeholders, Sessions, feed_dictionaties, etc.

### First adoption of eager execution by TF -> 2017
https://medium.com/@yaroslavvb/tensorflow-meets-pytorch-with-eager-mode-714cce161e6c
```python
pip install tf-nightly-gpu
python
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
a = tf.random_uniform((10,))
b = tf.random_uniform((10,))
for i in range(100):
  a = a*a
  if a[0]>b[0]:
  break
print(i)
```

### Recommended High-Level API
In TF 2.0, tf.keras is the recommended high-level API. And there are no Low-Level guide in TF 2.0 (https://github.com/tensorflow/tensorflow/issues/33823)

### Possible ways of Creating a Model in Keras tf 2.0
source: https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
Inside of this tutorial you’ll learn how to utilize each of these methods, including how to choose the right API for the job.

Keras and TensorFlow 2.0 provide you with three methods to implement your own neural network architectures:

- <b>Sequential API</b>: allows you to create models layer-by-layer in a step-by-step fashion.
    Easiest but also most limited — you cannot create models that:
    - Share layers
    - Have branches (at least not easily)
    - Have multiple inputs
    - Have multiple outputs
- <b>Functional API</b>: can create all that is possible with Sequantial API and more like:
    - Create more complex models.
    - Have multiple inputs and multiple outputs.
    - Easily define branches in your architectures (ex., an Inception block, ResNet block, etc.).
    - Design directed acyclic graphs (DAGs).
    - Easily share layers inside the architecture.
- <b>Model subclassing</b>: Model subclassing is fully-customizable and enables you to implement your own custom forward-pass of the model. However, this flexibility and customization comes at a cost — model subclassing is way harder to utilize than the Sequential API or Functional API.

### TensorFlow 2.0 -> Functions not session
source: https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md

Comparing TensorFlow code today with how we propose it looks in 2.x:


<table>
  <tr>
   <td>TensorFlow 1.x
   </td>
   <td>2.0
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">W = tf.Variable(
  tf.glorot_uniform_initializer()(
    (10, 10)))
b = tf.Variable(tf.zeros(10))
c = tf.Variable(0)

x = tf.placeholder(tf.float32)
ctr = c.assign_add(1)
with tf.control_dependencies([ctr]):
  y = tf.matmul(x, W) + b
init = 
  tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  print(sess.run(y,
  feed_dict={x: make_input_value()}))
  assert int(sess.run(c)) == 1</pre>


   </td>
   <td>



<pre class="prettyprint">W = tf.Variable(
  tf.glorot_uniform_initializer()(
    (10, 10)))
b = tf.Variable(tf.zeros(10))
c = tf.Variable(0)

@tf.function
def f(x):
  c.assign_add(1)
  return tf.matmul(x, W) + b

print(f(make_input_value())
assert int(c) == 1</pre>


   </td>
  </tr>
</table>

### Is low level worth it according to tf2? - official source

source: https://www.tensorflow.org/guide/migrate

Worthy of note here - in TensorFlow 1.x, the memory underlying the variables `W` and `b` in the runtime lives for the lifetime of the `Session` - unrelated to the lifetime of the Python objects. In 2.x, the lifetime of the Python objects and the runtime state are tied together.


3. Upgrade your training loops

Use the highest level API that works for your use case. Prefer tf.keras.Model.fit over building your own training loops.

These high level functions manage a lot of the low-level details that might be easy to miss if you write your own training loop. For example, they automatically collect the regularization losses, and set the training=True argument when calling the model.

### What's a tensor?

source: https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/#tensors-the-basic

Every tensor has a name, a type, a rank and a shape.

- The name uniquely identifies the tensor in the computational graphs (for a complete understanding of the importance of the tensor name and how the full name of a tensor is defined, I suggest the reading of the article Understanding Tensorflow using Go).

- The type is the data type of the tensor, e.g.: a tf.float32, a tf.int64, a tf.string, …

- The rank, in the Tensorflow world (that’s different from the mathematics world), is just the number of dimension of a tensor, e.g.: a scalar has rank 0, a vector has rank 1, …

- The shape is the number of elements in each dimension, e.g.: a scalar has a rank 0 and an empty shape (), a vector has rank 1 and a shape of (D0), a matrix has rank 2 and a shape of (D0, D1) and so on.

So you might wonder: what’s difficult about the shape of a tensor? It just looks easy, is the number of elements in each dimension, hence we can have a shape of () and be sure to work with a scalar, a shape of (10) and be sure to work with a vector of size 10, a shape of (10,2) and be sure to work with a matrix with 10 rows and 2 columns. Where’s the difficulty?

#### Tensor’s shape

The difficulties (and the cool stuff) arises when we dive deep into the Tensorflow peculiarities, and we find out that there’s no constraint about the definition of the shape of a tensor. Tensorflow, in fact, allows us to represent the shape of a Tensor in 3 different ways:

- Fully-known shape: that are exactly the examples described above, in which we know the rank and the size for each dimension.
- Partially-known shape: in this case, we know the rank, but we have an unknown size for one or more dimension (everyone that has trained a model in batch is aware of this, when we define the input we just specify the feature vector shape, letting the batch dimension set to None, e.g.: (None, 28, 28, 1).
- Unknown shape and known rank: in this case we know the rank of the tensor, but we don’t know any of the dimension value, e.g.: (None, None, None).
- Unknown shape and rank: this is the toughest case, in which we know nothing about the tensor; the rank nor the value of any dimension.

Tensorflow, when used in its non-eager mode, separates the graph definition from the graph execution. This allows us to first define the relationships among nodes and only after executing the graph.