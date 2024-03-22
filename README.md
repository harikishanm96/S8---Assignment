# S8---Assignment

## Accuracy

GN:
Train	- 71.17,	
Test	- 74.09

LN:
Train	- 71.37,
Test	- 74.95

BN:
Train	- 71.72,
Test	- 74.18

## Code

Convolutional Layers: The network consists of several convolutional layers (nn.Conv2d). These layers perform the convolution operation, applying learnable filters to input images to extract features.
Example: nn.Conv2d(3, 8, 3, padding=1, bias=False) creates a convolutional layer with 3 input channels, 8 output channels (number of filters), a kernel size of 3x3, and padding of 1 pixel.

Activation Functions: Rectified Linear Unit (ReLU) activation functions (nn.ReLU()) are used after each convolutional layer to introduce non-linearity into the network.
Example: nn.ReLU() applies ReLU activation element-wise to the output of the previous layer.
Normalization Layers:

Normalization layers are used to stabilize and speed up the training process. This network supports three types of normalization:
Batch Normalization (nn.BatchNorm2d): Normalizes activations across the batch dimension.
Layer Normalization (nn.GroupNorm(1, n)): Normalizes activations across the channel dimension.
Group Normalization (nn.GroupNorm(num_groups, n)): Normalizes activations across a specified number of groups.
Example: nn.BatchNorm2d(n) creates a batch normalization layer with n channels.

Dropout: Dropout layers (nn.Dropout2d) are used to prevent overfitting by randomly setting a fraction of input units to zero during training.
Example: nn.Dropout2d(drop) applies 2D dropout with a specified dropout probability.

Pooling Layers: Max pooling layers (nn.MaxPool2d) are used to downsample the spatial dimensions of the input volume.
Example: nn.MaxPool2d(2, 2) performs max pooling with a kernel size of 2x2 and a stride of 2.

Fully Connected Layer: The output layer consists of an average pooling operation followed by a 1x1 convolutional layer to produce class scores.
Example: nn.Conv2d(in_channels=48, out_channels=10, kernel_size=(1, 1), bias=False) creates a convolutional layer with 48 input channels and 10 output channels, effectively reducing the spatial dimensions to 1x1.

Forward Pass: The forward method defines how input data flows through the layers of the network.
It applies convolutional, activation, normalization, dropout, and pooling layers sequentially to the input data, followed by the output layer.
Finally, it applies a softmax function to the output to obtain class probabilities.

## Normalization techniques

Batch Normalization (BN): Batch normalization (nn.BatchNorm2d) normalizes the activations of each layer across the batch dimension.
It computes the mean and variance of each feature map over the batch and applies a linear transformation to normalize the activations.
Batch normalization helps to reduce internal covariate shift, making training more stable and allowing higher learning rates.
It has been widely used in deep learning architectures and has shown to improve convergence and generalization.
Example usage: nn.BatchNorm2d(n)

Layer Normalization (LN): Layer normalization (nn.GroupNorm(1, n)) normalizes the activations of each layer across the channel dimension.
It computes the mean and variance of each feature map across channels and applies normalization independently for each feature map.
Layer normalization is less sensitive to the batch size and can be used effectively even with small batch sizes.
It can improve training stability and performance, especially in recurrent neural networks (RNNs) and transformers.
Example usage: nn.GroupNorm(1, n)

Group Normalization (GN): Group normalization (nn.GroupNorm(num_groups, n)) divides the channels into groups and computes normalization statistics separately for each group.
It aims to strike a balance between the computation of batch normalization and the robustness of layer normalization.
Group normalization can be beneficial when the batch size is small or when the feature maps have irregular channel dimensions.
It has been observed to perform well in scenarios where batch sizes are limited, such as transfer learning or fine-tuning tasks.
Example usage: nn.GroupNorm(num_groups, n)


