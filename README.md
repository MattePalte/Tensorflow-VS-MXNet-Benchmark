# Benchmark of TensorFlow 2.1 and MXNet 1.5

This repo is organized in the following folders:
- Literature: official papers about the two frameworks published by their creators (i.e. Google and Amazon) plus some related work in terms of benchmark
- Developement Notebook: it contains various Python notebooks used to test various ideas and networks
- Scripts: contains the final version of code snippets used to reproduce the experiments
- Results on GPU machine: it contains all the generated files produced during the Benchmark A (fundamental operations) and Benchmark C (optimization hybridize vs XLA)
- Discussion: some markdown file that we used in the team to share findings and organize the work

# Benchmarks contained

In the context of Big Data Analytics Project, the following benchmarks were carried out under the supervision of Behrouz Derakhshan:
Benchmark A: Benchmarking of low level fundamental operations on both GPU and CPU
Benchmark B: End-to-End benchmark of LeNet Convolutional Neural Network training on CPU and GPU
Benchmark C: End-to-end benchmark of XLA and Hybridised on LeNet on CPU and GPU
Benchmark D: Benchmark of TensorFlow optimizations: eager execution and XLA.

For a better description of consult the report.pdf file in the main folder

# How to reproduce the benchmarks

## Requirements
- PC with GPU with computational capabilities 5.2 or more
- Instal Nvidia Grapich Card & Drivers
- Install compatible CUDA drivers
- Install compatible cuDNN drivers
- Check your newly installed drivers with “nvidia-smi” command
- Install TensorFlow 2.1 via pip install tensorflow==2.1
- Instal MXNet 1.5 via pip install mxnet==1.5.2

## Benchmark A: Fundamental Operations

1. Go to folder scripts
2. Launch jupyter notebbok
3. Open the notebook "Benchmark A"
4. follow the instuctions there

Optional: to visualize data with a quantile visualization move you result in the "Results on GPU machine" folder and use the notebook "Visualize Fundam Ops Quantile Trend" that is there

## Benchmark B: LeNet - End-to-End

1. Go to folder Developement Notebooks
2. Launch jupyter notebbok
3. Open the notebook "Benchmark B and D - LeNet  - Tensorflow with Eagerexecution OnOff - Final "
4. Open the notebook "Benchmark B - LeNet  - MXNet - Final "
5. Follow the instuctions there, or you can simply click on "Kernal", then "Restart and Run All"

## Benchmark C: Hybridize vs XLA
1. Go to folder scripts
2. Launch jupyter notebbok
3. Open the notebook "Benchmark C"
4. follow the instuctions there

## Benchmark D: Tensorflow Optimizations

1. Go to folder Developement Notebooks
2. Launch jupyter notebbok
3. Open the notebook "Benchmark B and D - LeNet  - Tensorflow with Eagerexecution OnOff - Final "
4. Open the notebook "Benchmark D - Classifying CIFAR-10 with XLA "
5. Follow the instuctions there, or you can simply click on "Kernal", then "Restart and Run All"

