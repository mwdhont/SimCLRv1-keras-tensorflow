# SimCLR

A Tensorflow-Keras Implementation of SimCLRv1 which allows to improve the feature representation quality of your base_model by the means of the Simple Framework for Contrastive Learning of Visual Representations. The provided code should require only minor changes in order to apply the framework to any Keras-model.

References: [Paper](https://arxiv.org/abs/2002.05709), [GitHub](https://github.com/google-research/simclr), [Blog](https://amitness.com/2020/03/illustrated-simclr/)


<p align="center">
  <img src="https://camo.githubusercontent.com/d92c0e914af70fe618cf3ea555e2da1737d84bc4/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966" alt="alt text" width="300"/>
  <figcaption align="center">Fig.1 - SimCLR Illustration <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html"> [1] </a> </figcaption>
</p>


# How to use?

SimCLR = SimCLR(base_model, input_shape, batch_size, feat_dim, feat_dims_ph, num_of_unfrozen_layers, save_path)

The method SimCLR.train can be used to train the SimCLR_model by passing the training and validation data of the type [DataGeneratorSimCLR](DataGeneratorSimCLR.py). The attribute SimCLR.base_model keeps track of the changing base_model. The feature representation quality can be evaluated in a number of ways, see below.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Npf8sE0dlyV0-SAISnsrGsJBjRDZM-EQ/view?usp=sharing)


# Implementation

A [SimCLR-class](SimCLR.py) has been defined which builds a Keras SimCLR_model around the base_model. It is the aim to improve the feature encoding quality of this base_model. The SimCLR_model has (*2.batch_size*) Inputs and 1 matrix-output with shape (*batch_size* x *4.batch_size*).
  * The *batch_size* images are each transformed twice by a random operation (see Fig.1), giving the 2.*batch_size* input images.
  * The output is computed with the help of a custom Keras-layer: [SoftmaxCosineSim](SoftmaxCosineSim.py) (see [notebook](0_illustration_SoftmaxCosineSim.ipynb) for intuitive toy example), in line with the NT-Xent-loss. The SimCLR-output matrix should match to [I|O|I|O], with I = identity-matrix and O = zero-matrix.
  * Consequently, a simple Keras cross_entropy-loss can be evaluated between the SimCLR-output and [I|O|I|O]. As such, the SimCLRmodel can be trained and simultaneously the feature encoding improves.

Difference with official [implementation](https://github.com/google-research/simclr):

  * Adam optimizer instead of Lars, no warmup nor cosine decay on learning rate, learning is reduced on plateau.
  * Reduced color_jitter strength to 0.5 instead of 1.0
  * As only 1 device is used, no global batch normalization
  * swish activation instead of relu in projection head

# Experiments

SimCLR has been used as a self-supervised learning approach to improve the feature encoding quality of a pretrained VGG16-network. A SimCLR_model has been build around the base model and consequently trained on the SimCLR-task. For this, a gradual defreeze of the base model was adopted. A clear improvement of the feature representations could be observed for the downstream classification task.

### Data: Trashnet

The [trashnet-dataset](https://github.com/garythung/trashnet) has been used.
The original dataset has been reduced to 5 classes with the following number of instances:
  * Glass: 501
  * Paper: 594
  * Cardboard: 403
  * Plastic: 482
  * Metal: 410

The original images of (512x384) have been center-cropped and reduced to a size (80x80)
Data has been split in train/val/test - 70/15/15.

### Evaluation

The feature quality is evaluated by the means of
  * A linear classifier (logistic regression) trained on the extracted features of the encoder
  * A fine-tuned classifier. 5 attempts are performed, the best classifier is kept.
  * A t-SNE visualization is made.

This is done for 3 fractions of the whole training data: 100%, 20%, 5%. It can be seen that the SimCLR improves the classification performance for all fractions of the training set.


### Results

Since the results change slightly due to the stochastic nature of the optimization procedure of both the SimCLR_model and the fine-tuned classifier, the average and standard deviation over 10 runs is presented below.

|   Fraction of training set   |  Classifier   | VGG16      |  SimCLR |
|:----------:|:-------------:|:-------------:|:------:|
| 100% | Linear | 0.79 ± 0.00 | 0.82 ± 0.01
|      | Fine-tuned | 0.84 ± 0.02| 0.87 ± 0.0
| 20% | Linear | 0.70 ± 0.0.| 0.81 ±0.02
|      | Fine-tuned | 0.84 ± 0.01| 0.86 ± 0.00
| 5% | Linear | 0.63 ± 0.00| 0.78 ± 0.05
|      | Fine-tuned | 0.79 ± 0.03| 0.82 ± 0.02


<img src=/img/t-SNE_VGG16.png alt="alt text" width="250"/>  |  <img src=/img/t-SNE_SimCLR.png alt="alt text" width="250"/>
:-------------------------:|:-------------------------:
Fig.2.1 - t-SNE of VGG16-features before SimCLR          | Fig.2.2 - t-SNE of VGG16-features after SimCLR



# Project Context

This repository is made in the context of a joined research project of [KU Leuven](https://www.kuleuven.be/kuleuven/), [Sagacify](https://sagacify.com/) and [BESIX](https://www.besix.com/en) on the topic of automatic monitoring of waste containers on construction site. For this purpose, data has been collected for the period of 5 months on a construction site. If you would be interested in the details of this research, please feel free to reach out.

<p align="center">

  <img src=/img/container1.png alt="alt text" width="300"/>
  <img src=/img/container2.png alt="alt text" width="300"/>
  <figcaption align="center">Fig.3 - Illustration of ContAIner output </figcaption>
</p>
