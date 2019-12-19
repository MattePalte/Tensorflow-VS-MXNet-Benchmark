# Benchmark

## How to profile in MXNet
https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html

The correct way to measure running time of MXNet models is to use MXNet profiler. In the rest of this tutorial, we will learn how to use the MXNet profiler to measure the running time and memory consumption of MXNet models. You can import the profiler and configure it from Python code.

```python
from mxnet import profiler

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')
# Ask the profiler to start recording
profiler.set_state('run')

run_training_iteration(*next(itr))

# Make sure all operations have completed
mxnet.nd.waitall()
# Ask the profiler to stop recording
profiler.set_state('stop')
# Dump all results to log file before download
profiler.dump()                
```

## How to profile in TF 2.0
https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras

This tutorial presents very basic examples to help you learn how to enable profiler when developing your Keras model. You will learn how to use the Keras TensorBoard callback to visualize profile result. Profiler APIs and Profiler Server mentioned in “Other ways for profiling” allow you to profile non-Keras TensorFlow job.
```python
# Load the TensorBoard notebook extension.
%load_ext tensorboard

log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

model.fit(train_data,
          steps_per_epoch=20,
          epochs=5, 
          callbacks=[tensorboard_callback])
```
```console    
root@kali:~ tensorboard --logdir=logs/profile/ --port=6006
```
Then access via http://localhost:6006/#profile
## Difference TF 1.14 and TF 2.0
https://github.com/tensorflow/tensorflow/issues/32104

## Reproducible Results
https://medium.com/datadriveninvestor/getting-reproducible-results-in-tensorflow-3705536aa185

## Past works

- Nvidia benchmark - Also pytorch and big datasets
https://developer.nvidia.com/deep-learning-performance-training-inference


