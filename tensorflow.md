## Major characteristic


### Eager Execution By Default
source: https://hackernoon.com/everything-you-need-to-know-about-tensorflow-2-0-b0856960c074 

TF 1.x : As you might recall, to build a Neural Net in TF 1.x, we needed to define this abstract data structure called a Graph. Also, (as you probably have tried), if we attempted to print one of the graph nodes, we would not see the values we were expecting. Instead, we would see a reference to the graph node. To actually, run the graph, we needed to use an encapsulation called a Session. And using the Session.run() method, we could pass Python data to the graph and actually train our models.

TF 2.0 : With eager execution, this changes. Now, TensorFlow code can be run like normal Python code. Eagerly. Meaning that operations are created and evaluated at once. TensorFlow 2.0 code looks a lot like NumPy code. In fact, TensorFlow and NumPy objects can easily be switched from one to the other. Hence, you do not need to worry about placeholders, Sessions, feed_dictionaties, etc.

### Recommended High-Level API
In TF 2.0, tf.keras is the recommended high-level API.