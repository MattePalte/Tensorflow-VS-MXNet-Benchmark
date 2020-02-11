# Tensorflow 1
W = tf.Variable(
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
  assert int(sess.run(c)) == 1

# Tensorflow 2
W = tf.Variable(
  tf.glorot_uniform_initializer()(
    (10, 10)))
b = tf.Variable(tf.zeros(10))
c = tf.Variable(0)

@tf.function
def f(x):
  c.assign_add(1)
  return tf.matmul(x, W) + b

print(f(make_input_value())
assert int(c) == 1

