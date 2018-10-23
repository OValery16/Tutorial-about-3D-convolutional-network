# Tutorial about 3D convolutional network

For me, the Artificial Intelligence is like a passion and I am trying to use it to solve some daily life problems. In this tutorial/project, I want to give some intuitions to the readers about how 3D convolutional neural networks are actually working. There are not a lot of tutorial about 3D convolutional neural networks, and not of a lot of them investigate the logic behind these networks.

## "Standard" convolutional network

Before to dive into 3D CNN, let's summarize together what we know about ConvNets. ConvNets consists mainly in 2 parts:
	
	* The feature extractor: this part of the network takes as input the image and extract the features that are meaningful for its classifcation. It amplify aspects of the input that are important for discrimination and suppress irrelevant variations. Usually, the feature extractor consists of several layers.	For instance, an image which could be seen as an array of pixel values. The first layer often learn representations that represent the presence or absence of edges at particular orientations and locations in the image. The second layer typically detects motifs by spotting particular arrangements of edges, regardless of small variations in the edge positions. Finaly the third layer may assemble motifs into larger combinations that correspond to parts of familiar objects, and subsequent layers would detect objects as combinations of these parts. 
	
	
    * The classifier: this part of the network takes as input the previous computed features and use them to predict the correct label.
	
![file architecture](/images/convolutional_neural_network.png?raw=true)
	
In order to extract such features, ConvNets use 2D convolution operations. 

![conv2D](/images/conv2D.gif)

## Why do we need a 3D CNN ?

Tranditionally, ConvNets are targeting RGB images (3 channels). The goal of 3D CNN is to take as input a video and extact features from it. When ConvNets extract the graphical characteristics of single image and put them in a vector (a low level representation), 3D CNNs extract the graphical characteristics of a **SET** of images. 3D CNNs takes in to account a temporal dimension (the order of the images in the video). From a set of images, 3D CNNs find a low level representation of a set of images, and this reprensentation is useful to find the right label of the video (a given action is performed)

In order to extract such feature, 3D convolution use 3Dconvolution operations. 

![conv3D](/images/conv3D.gif)