# Multi Layers Perceptron

# TF - Keras - MLP for binary classification:
# https://keras.io/getting-started/sequential-model-guide/#multilayer-perceptron-mlp-for-multi-class-softmax-classification

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Get your data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Configures the model for training
model.compile(loss='binary_crossentropy',
              optimizer = optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

# Start the training loop
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)




# MXnet - Gluon - MLP for binary classification:
# https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/training/fit_api_tutorial.html
# https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.contrib.estimator import estimator

# Get your data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
dataset = mx.gluon.data.dataset.ArrayDataset(x_train, y_train)
train_data_loader = gluon.data.DataLoader(dataset, batch_size=128)

# Build the model
model = gluon.nn.Sequential()
with model.name_scope():
    model.add(gluon.nn.Dense(64, activation="relu"))
    model.add(gluon.nn.Dense(64, activation="relu"))
    model.add(gluon.nn.Dense(1, activation="sigmoid"))

# Configures the model for training
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})
train_acc = mx.metric.Accuracy() # Metric to monitor

# Define the estimator, by passing to it 
# the model, loss function, metrics, trainer object and context
est = estimator.Estimator(net = model,
                          loss = loss_fn,
                          metrics = train_acc,
                          trainer = trainer)

# Start the training loop
est.fit(train_data = train_data_loader, epochs=20)