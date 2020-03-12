import os
import sys
import numpy as np
import gzip
import pandas as pd
import sys, getopt
from time import time
import pickle
print("OS: ", sys.platform)
print("Python: ", sys.version)
# MXnet
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
# Tensorflow
from sklearn.model_selection import train_test_split
# others
from generator import *
from datetime import datetime



import random
import json


def main(argv):
    mode = 'cpu'
    verbose = False
    xla = False
    hybridize = False
    epochs = 10
    tf_enable = True
    mx_enable = True
    try:
        opts, args = getopt.getopt(argv,"hm:v:e:x:p:t:n",["mode=","verbose=","epochs=", "xla=", "hybridize=", "tensorflow=", "mxnet="])
    except getopt.GetoptError:
        print('test.py -m <cpu or gpu> -verbose <true or false> -e <nr epochs> -x <true = activate XLA> -p <true = activate Hybridize>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -m <cpu or gpu> -verbose <true or false> -e <nr epochs> -x <true = activate XLA> -p <true = activate Hybridize>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-t", "--tensorflow"):
            tf_enable = (arg == "true")
        elif opt in ("-n", "--mxnet"):
            mx_enable = (arg == "true")
        elif opt in ("-x", "--xla"):
            xla = (arg == "true")
        elif opt in ("-p", "--hybridize"):
            hybridize = (arg == "true")
        elif opt in ("-v", "--verbose"):
            verbose = (arg == "true")
    print('Mode: ', mode)
    print('Verbose: ', verbose)
    print('Test TensorFlow', tf_enable)
    print('Test MXnet', mx_enable)
    print('Perform also XLA: ', xla)
    print('Perform also Hybridize: ', hybridize)

    # GPU vs CPU
    if mode == "gpu":
        # set the connect for MXNet
        mx_context = mx.gpu()
        mx.test_utils.list_gpus()
    else:
        mx_context = mx.cpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    
    import tensorflow as tf
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    
    print("Using the current versions. TF: ", tf.__version__, " - MX :", mx.__version__)
    
    print("Tensorflow - GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    #### AUXILIARY FUNCTIONS
    def read_mnist(images_path: str, labels_path: str):
        #mnist_path = "data/mnist/"
        #images_path = mnist_path + images_path
        print(images_path)
        with gzip.open(labels_path, 'rb') as labelsFile:
            labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path,'rb') as imagesFile:
            length = len(labels)
            # Load flat 28x28 px images (784 px), and convert them to 28x28 px
            features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                            .reshape(length, 784) \
                            .reshape(length, 28, 28, 1)
        return features, labels

    def get_mx_lenet():
        # MXNET -> GLUON
        # IDENTICAL TO LeNet paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf 
        model_mx = nn.HybridSequential()
        model_mx.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10))
        return model_mx

    def get_tf_lenet():
        # TENSORFLOW -> KERAS
        model_tf = keras.Sequential()
        init_tf = tf.keras.initializers.GlorotNormal(seed=1)
        model_tf.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1), kernel_initializer = init_tf, bias_initializer = init_tf))
        model_tf.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
        model_tf.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_initializer = init_tf, bias_initializer = init_tf))
        model_tf.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
        model_tf.add(layers.Flatten())
        model_tf.add(layers.Dense(units=120, activation='relu', kernel_initializer = init_tf, bias_initializer = init_tf))
        model_tf.add(layers.Dense(units=84, activation='relu', kernel_initializer = init_tf, bias_initializer = init_tf))
        model_tf.add(layers.Dense(units=10, activation = 'softmax', kernel_initializer = init_tf, bias_initializer = init_tf))
        return model_tf

    # MXNET
    def mx_training_procedure(handwritten_net, train_data, epochs, ctx):
        handwritten_net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
        #handwritten_net(init = mx.init.Xavier(), ctx=ctx)
        optim = mx.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, lazy_update=True)
        trainer = gluon.Trainer(handwritten_net.collect_params(), optim)
        # Use Accuracy as the evaluation metric.
        metric = mx.metric.Accuracy()
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

        for i in range(epochs):
            # Reset the train data iterator.
            train_data.reset()
            # Loop over the train data iterator.
            for batch in train_data:
                # Splits train data into multiple slices along batch_axis
                # and copy each slice into a context.
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                # Splits train labels into multiple slices along batch_axis
                # and copy each slice into a context.
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                # Inside training scope
                with autograd.record():
                    for x, y in zip(data, label):
                        z = handwritten_net(x)
                        # Computes softmax cross entropy loss.
                        loss = softmax_cross_entropy_loss(z, y)
                        # Backpropogate the error for one iteration.
                        loss.backward()
                        outputs.append(z)
                # Updates internal evaluation
                metric.update(label, outputs)
                # Make one step of parameter update. Trainer needs to know the
                # batch size of data to normalize the gradient by 1/batch_size.
                trainer.step(batch.data[0].shape[0])
            # Gets the evaluation result.
            name, acc = metric.get()
            # Reset evaluation result to initial state.
            metric.reset()
            print('training acc at epoch %d: %s=%f'%(i, name, acc))
        return handwritten_net

    def tf_training_procedure(model_tf, train_generator_tf, validation_generator_tf, epochs, batch, nr_train, nr_vali):
        # TENSORFLOW
        chosen_tf_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model_tf.compile(loss=keras.losses.categorical_crossentropy, optimizer=chosen_tf_optimizer, metrics=['accuracy'])
        steps_per_epoch = nr_train//batch
        validation_steps = nr_vali//batch
        model_tf.fit(train_generator_tf, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                            validation_data=validation_generator_tf, validation_steps=validation_steps, 
                            shuffle=True, callbacks=[])
        return model_tf


    def mx_get_accuracy(model_mx, test_data_mx, ctx):
        # TEST THE NETWORK
        metric = mx.metric.Accuracy()
        # Reset the test data iterator.
        test_data_mx.reset()
        # Loop over the test data iterator.
        for batch in test_data_mx:
            # Splits test data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            # Splits validation label into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(model_mx(x))
            # Updates internal evaluation
            metric.update(label, outputs)
        print('MXnet - Test %s : %f'%metric.get())
        return metric.get()

    def tf_get_accuracy(model_tf, test):
        # TENSORFLOW
        score = model_tf.evaluate(test['features'], to_categorical(test['labels']), verbose=0)
        print('TensorFlow - Test accuracy:', score[1])
        return score
    
    set_context(mx_context)

    # verbose
    if verbose:
        tf.debugging.set_log_device_placement(True)

    # reproducibility
    np.random.seed(42)
    random.seed(42)
    ctx = [mx_context]
    mx.random.seed(42)
    tf.random.set_seed(42)

    # LOAD TRAIN AND TEST ALREADY SPLIT
    train = {}
    test = {}
    train['features'], train['labels'] = read_mnist('lenet_data/train-images-idx3-ubyte.gz', 'lenet_data/train-labels-idx1-ubyte.gz')
    test['features'], test['labels'] = read_mnist('lenet_data/t10k-images-idx3-ubyte.gz', 'lenet_data/t10k-labels-idx1-ubyte.gz')
    print(test['features'].shape[0], '-> # of test images.')
    print(train['features'].shape[0], '-> # of training images (train + validation).')
    # CREATE TRAIN AND VALIDATION SPLIT
    validation = {}
    train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)
    print("    ", train['features'].shape[0], '-> # of (actual) training images.')
    print("    ", validation['features'].shape[0], '-> # of validation images.')

    # GENERAL PARAMETERS
    EPOCHS = epochs
    BATCH_SIZE = 200

    # MXNET
    # convert from NHWC to NCHW that is used by MXNET
    # https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
    X_train_mx = mx.ndarray.transpose(mx.nd.array(train['features']), axes=(0, 3, 1, 2))
    y_train_mx = mx.nd.array(train['labels'])
    X_validation_mx = mx.ndarray.transpose(mx.nd.array(validation['features']), axes=(0, 3, 1, 2))
    y_validation_mx = mx.nd.array(validation['labels'])
    X_test_mx = mx.ndarray.transpose(mx.nd.array(test['features']), axes=(0, 3, 1, 2))
    y_test_mx = mx.nd.array(test['labels'])
    # create data iterator
    train_data_mx = mx.io.NDArrayIter(X_train_mx.asnumpy(), y_train_mx.asnumpy(), BATCH_SIZE, shuffle=True)
    val_data_mx = mx.io.NDArrayIter(X_validation_mx.asnumpy(), y_validation_mx.asnumpy(), BATCH_SIZE)
    test_data_mx = mx.io.NDArrayIter(X_test_mx.asnumpy(), y_test_mx.asnumpy(), BATCH_SIZE)

    # TENSORFLOW
    # convert in multiple output for tensorflow
    X_train_tf, y_train_tf = train['features'], to_categorical(train['labels'])
    X_validation_tf, y_validation_tf = validation['features'], to_categorical(validation['labels'])
    # create data generator
    train_generator_tf = ImageDataGenerator().flow(X_train_tf, y_train_tf, batch_size=BATCH_SIZE)
    validation_generator_tf = ImageDataGenerator().flow(X_validation_tf, y_validation_tf, batch_size=BATCH_SIZE)
    
    output_folder = "your_benchmark_result_hybridize_vs_xla"
    print("Saving folder: ", output_folder)

    if mx_enable:
        # reproducibility
        np.random.seed(42)
        random.seed(42)
        ctx = [mx_context]
        mx.random.seed(42)
        tf.random.set_seed(42)
        # REINITIALIZE THE NETWORKS
        model_mx = get_mx_lenet()    
        # VANILLA MXNET 
        start = time.process_time()
        trained_model_mx = mx_training_procedure(model_mx, train_data_mx, EPOCHS, ctx)
        end = time.process_time()
        elapsed_mx = end - start
        accuracy_mx = mx_get_accuracy(trained_model_mx, test_data_mx, ctx)
        ts = datetime.timestamp(datetime.now()) # current timestamp
        result_tf = { "framework": "MX", "runtime": mode,"epochs": EPOCHS, "time": elapsed_mx, "accuracy": str(accuracy_mx[1]), "optimization": False}
        json.dump(result_tf, open( output_folder + "/mx-vanilla."+mode+"."+str(ts)+".result", "w" ) )

        if (hybridize):
            # reproducibility
            np.random.seed(42)
            random.seed(42)
            ctx = [mx_context]
            mx.random.seed(42)
            tf.random.set_seed(42)
            # REINITIALIZE THE NETWORKS
            model_mx = get_mx_lenet()
            model_mx.hybridize()
            # OPTIMIZED MXNET 
            start = time.process_time()
            trained_model_mx = mx_training_procedure(model_mx, train_data_mx, EPOCHS, ctx)
            end = time.process_time()
            elapsed_mx = end - start
            accuracy_mx = mx_get_accuracy(trained_model_mx, test_data_mx, ctx)
            ts = datetime.timestamp(datetime.now()) # current timestamp
            result_tf = { "framework": "MX", "runtime": mode,"epochs": EPOCHS, "time": elapsed_mx, "accuracy": str(accuracy_mx[1]), "optimization": True}
            json.dump(result_tf, open( output_folder + "/mx-hybridize."+mode+"."+str(ts)+".result", "w" ) )
    
    if tf_enable:
        # reproducibility
        np.random.seed(42)
        random.seed(42)
        ctx = [mx_context]
        mx.random.seed(42)
        tf.random.set_seed(42)
        # REINITIALIZE THE NETWORKS
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)
        model_tf = get_tf_lenet()
        tf.config.optimizer.set_jit(False)
        # VANILLA TENSORFLOW
        start = time.process_time()
        trained_model_tf = tf_training_procedure(model_tf, train_generator_tf, validation_generator_tf, EPOCHS, BATCH_SIZE, X_train_tf.shape[0], X_validation_tf.shape[0])
        end = time.process_time()
        elapsed_tf = end - start
        accuracy_tf = tf_get_accuracy(trained_model_tf, test)
        ts = datetime.timestamp(datetime.now()) # current timestamp
        result_tf = { "framework": "TF", "runtime": mode, "epochs": EPOCHS, "time": elapsed_tf, "accuracy": str(accuracy_tf[1]), "optimization": False}
        json.dump(result_tf, open( output_folder + "/tf-vanilla."+mode+"."+str(ts)+".result", "w" ) )


        if(xla):
            # reproducibility
            np.random.seed(42)
            random.seed(42)
            ctx = [mx_context]
            mx.random.seed(42)
            tf.random.set_seed(42)
            # REINITIALIZE THE NETWORKS
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_jit(True)
            model_tf = get_tf_lenet()
            tf.config.optimizer.set_jit(True)
            # OPTIMIZED TENSORFLOW
            start = time.process_time()
            trained_model_tf = tf_training_procedure(model_tf, train_generator_tf, validation_generator_tf, EPOCHS, BATCH_SIZE, X_train_tf.shape[0], X_validation_tf.shape[0])
            end = time.process_time()
            elapsed_tf = end - start
            accuracy_tf = tf_get_accuracy(trained_model_tf, test)
            ts = datetime.timestamp(datetime.now()) # current timestamp
            result_tf = { "framework": "TF", "runtime": mode,"epochs": EPOCHS, "time": elapsed_tf, "accuracy": str(accuracy_tf[1]), "optimization": True}
            json.dump(result_tf, open( output_folder + "/tf-xla."+mode+"."+str(ts)+".result", "w" ) )

if __name__ == "__main__":
   main(sys.argv[1:])