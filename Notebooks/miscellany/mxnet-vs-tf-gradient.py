# MXNet -> Autograd
# https://mxnet.apache.org/api/python/docs/tutorials/getting-started/crash-course/3-autograd.html

from mxnet import nd
from mxnet import autograd

x = nd.array([[1, 2], [3, 4]])
# tell what to monitor
x.attach_grad()
# record operation
with autograd.record():
    # do the operation you want
    y = 2 * x * x
# compute backward pass
y.backward()
# derivative is stored in: 
x.grad

# TF -> GradientTape
# https://www.tensorflow.org/tutorials/customization/autodiff

import tensorflow as tf

x = tf.constant([[1, 2], [3, 4]]) 
# record the tape
with tf.GradientTape() as tape:
    # tell what to monitor
    tape.watch(x)
    # do the operation you want
    y = 2 * x * x
# read the gradient from the tape 
dy_dx = tape.gradient(y, x)

# Constant and Variable behaviour in differentiation
# https://stackoverflow.com/questions/44745855/tensorflow-variables-and-constants
