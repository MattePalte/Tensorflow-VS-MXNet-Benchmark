# Tensorflow-VS-MXNet-Benchmark
Comparison of Tensorflow and MXNet framworks using state-of-the-art networks. Some criteria such as training and inference cost, GPU and CPU performance, and software features (API and ...).

# Goal - Meeting date: 17 December

## Presentation: 20 min
- 2 min - Very brief intro to deep learning neural network
- 3 min - Introduce the Big Picture - tensor and mxnet. linear algebra for DL
Find original paper (vision paper). What's the overall architectecutre of the system (tf -> DAG). Mxnet (parameter severs style?). Which scheduling do they do?
- 4 min - Api differences -> comparison
- 4 min -> benchmarking results -> What we have done 
- 5 min what is the plan -> digging deeper into system
more complex network (Recurrent?)

## Next steps 
1. Read first paper of MXNet and Tensorflow
1. Capture the result of LeNet -> 6 million different verion of mnist (data augmentation is also possible)
1. Explore low level linear algebra (do some benchmarking)

## Additional notes/suggestions
- Try running differnet ways of dot product -> emprical result which api is better (random matrices, dot product)
- Benchmark code-> for normal users like me 
- Show two lenet 

# Goal - Meeting date: 4 December
Abstraction |Tensorflow | MXNet
-------------|--------------|--------------
High Level | e.g. Keras | e.g. Gluon
Low Level | variable API | MXNet Library

### Things to pay attention to:
1. Use same random seed (if possible)
1. Fix the same initialization (in terms of weight)
1. Same hyperparameters (e.g. optimizer, nr epochs, ...)
1. The accuracy should be the same (since the operations must be the same)
1. If not possible to enforce same random seed, run the train multiple times and take the average

### Caracteristics to compare:
The focus is on the train part only.
1. CPU (train time)
1. GPU - if available (train time)
1. Memory usage 
1. nr lines of code
1. readability


