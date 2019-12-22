## Major characteristic


### Eager Execution By Default
source: https://hackernoon.com/everything-you-need-to-know-about-tensorflow-2-0-b0856960c074 

TF 1.x : As you might recall, to build a Neural Net in TF 1.x, we needed to define this abstract data structure called a Graph. Also, (as you probably have tried), if we attempted to print one of the graph nodes, we would not see the values we were expecting. Instead, we would see a reference to the graph node. To actually, run the graph, we needed to use an encapsulation called a Session. And using the Session.run() method, we could pass Python data to the graph and actually train our models.

TF 2.0 : With eager execution, this changes. Now, TensorFlow code can be run like normal Python code. Eagerly. Meaning that operations are created and evaluated at once. TensorFlow 2.0 code looks a lot like NumPy code. In fact, TensorFlow and NumPy objects can easily be switched from one to the other. Hence, you do not need to worry about placeholders, Sessions, feed_dictionaties, etc.

### Recommended High-Level API
In TF 2.0, tf.keras is the recommended high-level API.

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


Worthy of note here - in TensorFlow 1.x, the memory underlying the variables `W` and `b` in the runtime lives for the lifetime of the `Session` - unrelated to the lifetime of the Python objects. In 2.x, the lifetime of the Python objects and the runtime state are tied together.
