# SimCLR
Tensorflow-Keras Implementation of SimCLRv1:
  * [Paper](https://arxiv.org/abs/2002.05709)
  * [GitHub](https://github.com/google-research/simclr)

An intuitive explanation can be found [here](https://amitness.com/2020/03/illustrated-simclr/).

<p align="center">
  <img src="https://camo.githubusercontent.com/d92c0e914af70fe618cf3ea555e2da1737d84bc4/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966" alt="alt text" width="300"/>
</p>


## Implementation

It has been chosen to define a SimCLR class which builds a Keras SimCLR_model around the base_model of which the feature encoding wants to be improved. The SimCLR_model has (2.batch_size) Inputs and 1 matrix-output with shape (batch_size x 4.batch_size). The output is computed with the help of a custom Keras-layer: SoftmaxCosineSim, in line with the NT-Xent-loss. The SimCLR-output should match to [I|O|I|O], so that a simple Keras cross_entropy-loss can be used to train the SimCLRmodel and consequently improve the feature encoding.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/2_model_SimCLR.ipynb)


Difference from official implementation:
  * Adam optimizer instead of Lars, no warmup and cosine decay on the learning rate, learning is reduced on plateau though.
  * Reduced color_jitter strength to 0.5 instead of 1.0
  * As only 1 device is used, no global batch normalization
  * swish activation instead of relu in projection head

## Experiment Data

The public available trashnet-dataset has been used: [GitHub repository](https://github.com/garythung/trashnet)
The original dataset has been reduced to 5 classes with the following number of instances:
  * Glass: 501
  * Paper: 594
  * Cardboard: 403
  * Plastic: 482
  * Metal: 410

The original images of (512x384) have been center-cropped and reduced to a size (80x80)
Data has been split in train/val/test - 70/15/15
Validation set is used for early stopping.

## Evaluation

The feature quality is evaluated by the means of
  * A linear classifier (logistic regression) trained on the extracted features of the encoder
  * A fine-tuned classifier. 5 attempts are given, results are subjective to the stochastic nature of the optimization procedure
  * a t-SNE visualization is made.

This for 3 fractions of the whole training data: 100%, 20%, 5%



## Results



|   Fraction of training data   |  Classifier   | VGG16      |  SimCLR |
|:----------:|:-------------:|:-------------:|:------:|
| 100% | Linear | 0.79 | 0.82
|      | Fine-tuned | 0.82 | 0.86
| 20% | Linear | 0.70 | 0.78
|      | Fine-tuned | 0.82 | 0.86
| 5% | Linear | 0.63 | 0.7
|      | Fine-tuned | 0.75 | 0.81




## Project Context

This repo is made in the context of a joined research project of [KU Leuven](https://www.kuleuven.be/kuleuven/), [Sagacify](https://sagacify.com/) and [BESIX](https://www.besix.com/en) on the topic of automatic monitoring of waste containers on construction site. For this purpose, data has been collected for the period of 5 months on site. If you would be interested in this research, please feel free to open an issue.


![alt text](/img/container1.png) ![alt text](/img/container2.png)