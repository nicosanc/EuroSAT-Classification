# Purpose

To showcase my understanding of a popular Deep Learning framework and the Deep Learning Basics,
I decided on building a Convolutional Neural Network to classify images, using PyTorch. I trained and tested the model 
on the CIFAR10 dataset offered by torchvision.datasets. The model was trained several times with different 
data augmentations to try and improve the model's generalization. It was tested against unshuffled, un-augmented validation and test 
sets. 

# How it was implemented

## Google Colab Notebook
Due to limitations in my personal computer's computing power, I decided to run this project on a Google Colab Notebook. Colab Notebooks are 
especially useful for testing and running large Data Science and Machine Learning projects by allowing you to remotely connect to Google's
GPUs and VPUs. This allows for much more efficient loading, training, and testing of large datasets. 
When I created this project, I used a VPU to accomplish fast training and testing. This VPU, along with some other GPUs available on the platform, can only be accessed through a paid membership. There still exists free access to GPUs of relatively lower computing power that can still accomplish the project's task.  

## The dataset
The CIFAR10 dataset included in the torch_visions library contains 60000 images, where each class has its own 6000 images. Training and testing sets are typically split 50000/10000. I used this ratio to split between training and hold out sets then split the 10000 images into 6500 for validation and 3500 for testing. The validation set is typically used for checking at the end of each epoch whether the model is overfitting and not generalizing well. With the validation set, we can make adjustments like updating hyperparameters and implementing regularization techniques. Once our model reaches a plateau in validation loss, we can consider it finalized and can now test it against our last hold out set. 

## PyTorch
the "torch" library in python includes numerous useful functionalities for developing image classifiers. PyTorch offers methods that can load datasets as iterable objects (DataLoader), transform images using data augmentations (transforms), define layers of the network (nn.Module), and even provide access to loss functions and optimizers. Having all these capabilites available through the same framework makes it much easier to build an end to end model, since the same documentation can be used to find relevant functions. PyTorch can also be used alongside other popular libraries like Numpy and Sci-kit Learn, which further expands what a project can accomplish.

## Model Architecture and Forward Pass
This model was designed with 6 convolutional layers, a max pool layer, 2 fully connected layers, and 2D batch normalization. The batch normalization steps were designed to each take in a specific number of output channels from convolutional layers, with the number growing as the layers increased in channel size. The max pool layers were applied after the first 3 convolutional layers and the last 3 convolutional layers. 
During the forward pass, every layer except the final fully connected layer went through a ReLU activation function. ReLU is a frequently used activation function that allows for the model to find non-linear relationships in the data. Most importantly, with much larger and deeper neural networks, it avoids a vanishing gradient to improve overall performance. Vanishing gradient comes from deep layers' gradients becoming so small as they get to the early layers that they basically stop learning anything new. This makes it difficult for large models to quickly converge to a good solution. 

# Plans for the future
To expand on the project and its capabilities, I plan on incorporating new models that can aid in image classification or even allow for new functionality. 
