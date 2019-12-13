# Benchmark

## How to profile in MXNet
https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html

The correct way to measure running time of MXNet models is to use MXNet profiler. In the rest of this tutorial, we will learn how to use the MXNet profiler to measure the running time and memory consumption of MXNet models. You can import the profiler and configure it from Python code.

```python
from mxnet import profiler
```