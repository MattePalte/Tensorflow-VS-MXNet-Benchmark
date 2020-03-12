# TF

Optimizations available:
 - eneble XLA 

 # MXNet

 Optimization available
 - .hybridixe "With MXNet/Gluon, calling .hybridize() on your network will cache the computation graph and you will get performance gains. However that means that you won't be able to step through every calculations anymore. Use it once you are done debugging your network.". 
    - Sources: https://github.com/ilkarman/DeepLearningFrameworks 
    - Official doc: https://beta.mxnet.io/guide/packages/gluon/hybridize.html
- convlutional layers: "cudnn_tune: enable this option leads to higher startup time but may give faster speed. Options are:
    - **off**: no tuning
    - **limited_workspace**:run test and pick the fastest algorithm that doesn't exceed workspace limit.
    - **fastest**: pick the fastest algorithm and ignore workspace limit.
    - **None** (default): the behavior is determined by environment variable ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace (default), 2 for fastest.
    - Source: https://mxnet.apache.org/api/python/docs/api/ndarray/ndarray.html?highlight=convolution#mxnet.ndarray.Convolution
