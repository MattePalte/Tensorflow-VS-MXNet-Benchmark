# Tensorflow-VS-MXNet-Benchmark
Comparison of Tensorflow and MXNet framworks using state-of-the-art networks. Some criteria such as training and inference cost, GPU and CPU performance, and software features (API and ...).

# Meeting 24 February

### Kaoutar findings:
- Model fit generator (introduced in 2.0) is faster than fit
- Eager execution introduced in 2.0 slowed down the execution
- TF has the best performance improvement in distributed setting
### Kaoutar Next steps
- compute by hand if the speedup in the train of acomplete nework is aligned with the fundamental operations
- Try easy setup/standard: for normal user
- Try advanced setup: XLA and disable eager generator

### Matteo findings:
- cProfile not so useful (the core operation are hidden)
- Mxnet is calling blas directly and memory in place operation (citation amazon paper 25 August 2019)
### Matteo Next steps:
- Analyze each of this optimization separately
- Analyze which is the dimensions of matrices in real world scenarios
- Try convolution op benchmark
- See if MXNet has technical optimizations not enable by default
- Try once with bigger dataset. ImageNet competition. At least once (basic vs fully optimized trend)

Q: can you explain the difference in runtime just with the difference in fundamental operations? -> quote the paper

### Slide
Select only some of the figure and explain the trend.

### Readme: 
- machine used
- package used
- How to setup the environment

# Meeting 28th January
- try run with GPU on the cluster

- LeNet (Kaoutar) -> run on CPU and GPU on CLUSTER
- Fundamental Ops (Matteo) -> run CPU and GPU on CLUSTER -> increment the dimensions of input (based on a realistic network)
- Other metrics (Matteo) -> (does it exists a light weight library) -> CPU/GPU usage and RAM -> useful to explain the results

- Reproduce the Paper's result -> with 2 dataset out of 4 (Kaoutar)
    - Handwritten digits classification with feed forward networks (MNIST) 
    - Image classification with convolutional networks (CIFAR-10)
    - (x) Sentiment analysis with LSTM Networks (IMDB)

Collect data from official sources

- Does it make sense to change network? or mxnet is clearly better from now?

REPORT -> Interpretation of results:
- if MXNET clearly better: Why is MXNet better? Convince the audience (TF employees) that our results are motivated. Which is the cause of this inefficiencies?
ACM Double Column paper -> Conference style (8 pages)
INTRO -> EVALUATION -> CONCLUSION (VERY IMORTANT) -> METASECTIONS (what we have done)
- bit of background of deep learning
- framworks
- details of each framework 
    - features wise comparison? (fundam operation lin alg, usability, emprical results) then summarize at the end of each feature
    - all tf first then all MXnet 
- discussion to summarize and gives our reasoning (why is it like that?)
- discribe other systems (pythorch, ) and other benchmark (rysia) cite different papers
- mentions how we come up with those info (blogs, authors of tf and mxnet, ) -> lot of references

CLUSTER:
- we will receive the 
- check the calendar
- critical (experiment important -> noone is using it)
- shared (can be used by someone else)

Ideas:
- reproduce one result of the paper at least for one dataset (also CIFAR dataset)
- start with lenet with a bigger dataset 
- try with Imagenet competiton newtorks ()
- pick a more recent and realistic network

# Goal: meeting date 13rd January

## Presentation: 20 min
- 2 min - Very brief intro to deep learning neural network
- 3 min - Introduce the Big Picture tools - tensor and mxnet. Historical differences. Define high level differences. Define if there are well know advantages one over the other. linear algebra for DL
Find original paper (vision paper). What's the overall architectecutre of the system (tf -> DAG). Mxnet (parameter severs style?). Which scheduling do they do? How is the distribution
- 4 min - Api differences -> comparison -> show few line of code in each. Show simple lin alg op and some simple networks (MLP).
if presenting lenet -> have to explain CNN how they work.
if RNN -> have
- 4 min - **microbenchmarking of fundamental operations** benchmarking results -> What we have done. Tips: run multiple times and take average and errorbar. Plot as funtion of input. 
- **run 10 epochs of lenet and compare results**
- 5 min what is the plan -> digging deeper into system

# Goal: meeting date 3rd January

## Presentation: 20 min
- 2 min - Very brief intro to deep learning neural network
- 3 min - Introduce the Big Picture - tensor and mxnet. linear algebra for DL
Find original paper (vision paper). What's the overall architectecutre of the system (tf -> DAG). Mxnet (parameter severs style?). Which scheduling do they do?
- 4 min - Api differences -> comparison
- 4 min - **microbenchmarking of fundamental operations** benchmarking results -> What we have done 
- **run 10 epochs of lenet and compare results**
- 5 min what is the plan -> digging deeper into system

Goal: give the students some valuable advice base on their needs (size of dataset, complexity of network etc) what framwork they should use?

### Assignment
 - complete microbenchmarking of fundamental operations
 - end-to-end network comparison

### DL Division in 
 - convolutional (LeNet)
 - lstm 
 - rnn
 - ffnn

for middterm -> good overview of microbenchmarking of fundamental doperations in the two frameworks
change name -> fundamental operations
add losses available in the two frameworks (cross entropy etc)

### General notes
- micro-benchmark - with random dat a in matrix multiplication -> how it scale, how fast do matrix, how much take to load csv file

- notebook %timeit cell,
dig deeper with python profiler if there are big differences

- have a look at the framework https://github.com/vdeuschle/rysia to investigate ways of profiling deep learning libraries in a fair and objective ways independent from the library itself

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


