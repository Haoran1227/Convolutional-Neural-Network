# Convolutional-Neural-Network
Implementation of Convolutional Neural Network without Tensorflow or other frames.

Haoran Zhao, 2019.11.30

Optimizers.py: Advanced optimization methods including Stochastic Gradient Descent(SGD), SGD with momentum and Adam optimizers.

FullyConnected.py: realizes a fully-connected layer with forward propagation, backward propagation and gradients update function.

Conv.py: realizes convolutional layer. The forward propagation has two achievements, downsampling after convolution and im2col.

Flatten.py: realizes flatten layer.

Pooling.py: realizes pooling layer.

ReLU.py: realizes a node with ReLU activation function.

SoftMax.py: realizes layer with softmax function for classification problem.

Loss.py: employs cross-entropy loss as loss function.

NeuralNetwork.py: builds a small neural network.

Helpers: Some functions which can help test neural network such as gradient-check and data generalization. 

NeuralNetworkTests.py: main function. It's used to test the validity of code.

NeuralNetworkTests_fullyconnected.py: It's used for the first part of exercise. Test of fully-connected neural network.
