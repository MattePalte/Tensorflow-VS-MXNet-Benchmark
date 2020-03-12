
# Matrix multiplication in real network

## Intro
General neural networks are composed of different types of layers:
- fully connected
- convolutional
- self attention

Every layer can be represented with a matrix multiplication.

## Fully connected
The output of a layer k are obtained by mutipling the weight matrix  times the output of the layer  k - 1 and then appling the activation function to the resulting vector. 

The dimensions involved are the following:
- previous layer (k-1) -> m neurons (row vector c1)
- current layer (k) -> n neurons (row vector c2)
- weights -> (m * n) (Matrix W)

The resulting operations are:
- c2 = activation(c1 * W)

Source: https://www.youtube.com/watch?v=lFOOjeH2wsY

## Convolution
The output is obtained by convolving a kernel/filter over the input matrix. In terms of matrix the most famous algorithm used is the im2col that convert the convolution operation in a matrix multiplication.

The dimension involved are the following:
- input matrix (usually an mage) -> n * n (matrix I1)
- kernel/filter (usally very small with respect to the image) -> m * m (matrix F)
- output matrix (feature map) -> k * k (dimension depending on the kernel type of operation)

The steps are the following:
1. compute the expected output dimension
1. apply 0 padding to your kernel matrix until it matches the expected output dimension
1. for each row of the resulting kernel-padded create a Toeplitz matrix Hi (coming from the i-th row)
1. compose this matrices together to obtain a doubly block Toeplitz matrix H
1. transform your input matrix I1 in a vector f by stacking all the rows vertically
1. compute the output vector g = H * f
1. apply activation function to vector g
1. apply the inverse operation to get I2 from vector g

The doubly block Toeplitz matrix H contains a number of block equal to the nr of entries in the input matrix (n * n) (i.e. the pixel in the input image).
Each of this block is composed of k * n. Therefore the final matrix H will have the dimension (n*k) * (n*n).

e.g an image 256 * 256 with a kernel of 5*5 will have a doubly block Toeplitz matrix H.
Supposing that each entry is an intensity value (0-255).

(256 * 5) * (256 * 256) * 4 bytes = 335.54432 megabytes

Matrix of int with dimension 1280 * 65536

Source: https://iccl.inf.tu-dresden.de/w/images/4/4e/Ml-11.pdf
Source: http://www.songho.ca/dsp/convolution/convolution2d_example.html

## Self attention
Usually this layer is used in the Natural Language Processing (NLP) communities and in one of their most popular model BERT, they are using a representation of each word that is called word embeddings (ususally it is a vector that is 256-dimensional, 512-dimensional, or 1024-dimensional).
In the larger version of BERT they use vector of 1024 to represent each word.

At each self-attention layer, each word vector has to be converted into three vecotrs: Query, Key, and Value (dimension 64). Therefore we need matrices with dimension: 1024 * 64.

Then a similarity score between each pair of word representation has to be computed, and it translates in the moltiplicatio of two matrices of dimension: max nr of word in a sequence * 64. Usaully this number of word is 512, therefore the final matrix will have dimension 512 * 64.

Source: http://jalammar.github.io/illustrated-transformer/
Source: http://jalammar.github.io/illustrated-bert/