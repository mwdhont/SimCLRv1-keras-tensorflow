# SimCLR

A Tensorflow-Keras Implementation of SimCLRv1 which allows to improve the feature representation quality of your base_model by the means of the Simple Framework for Contrastive Learning of Visual Representations (SimCLR). The provided code should allow to apply the framework to any Keras model with only minor changes.

<p align="center">
  <img src="https://camo.githubusercontent.com/d92c0e914af70fe618cf3ea555e2da1737d84bc4/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966" alt="alt text" width="300"/>
  <br>
  <m>Fig.1 - SimCLR Illustration <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html"> [1] </a> </m>
</p>

The given implementation allowed for an top-1 accuracy increase of 17% on the linear classifier trained, with 5% of the data. Furthermore, the t-SNE plot demonstrates a clear clustering of the features according to their class, after training with the SimCLR framework.

<img src=/img/t-SNE_VGG16.png alt="alt text" width="250"/>  |  <img src=/img/t-SNE_SimCLR.png alt="alt text" width="250"/>
:-------------------------:|:-------------------------:
Fig.2.1 - t-SNE of VGG16-features before SimCLR          | Fig.2.2 - t-SNE of VGG16-features after SimCLR

</p>

It is possible to reproduce this results via the following notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Npf8sE0dlyV0-SAISnsrGsJBjRDZM-EQ/view?usp=sharing)


References: [Paper](https://arxiv.org/abs/2002.05709), [GitHub](https://github.com/google-research/simclr), [Blog](https://amitness.com/2020/03/illustrated-simclr/)


# How to use?

SimCLR = SimCLR(base_model, input_shape, batch_size, feat_dim, feat_dims_ph, num_of_unfrozen_layers, save_path)

The method SimCLR.train can be used to train the SimCLR_model by passing the training and validation data of the type [DataGeneratorSimCLR](DataGeneratorSimCLR.py). The attribute SimCLR.base_model keeps track of the changing base_model. The feature representation quality can be evaluated in a number of ways, see below.

# Implementation

A [SimCLR-class](SimCLR.py) has been defined which builds a Keras SimCLR_model around the base_model. It is the aim to improve the feature encoding quality of this base_model. The SimCLR_model has (*2.batch_size*) Inputs of the image size and 1 matrix-output with shape (*batch_size* x *4.batch_size*).
  1. Each of the *batch_size* images are transformed twice by a random image distortion (see Fig.1), giving the 2.*batch_size* input images. See [DataGeneratorSimCLR](DataGeneratorSimCLR.py) and [SimCLR_data_util](SimCLR_data_util.py) for the details.
  2. These input images are passed through the base model and a MLP projection head, resulting in a feature encoding.
  3. The SimCLR_model-output is obtained from a pairwise vector multiplication between all computed feature encodings. This vector multiplications correspond with the cosine similarity, after which the similarity is passed through a softmax. Since it is the aim to 'attract' feature representations of the same image, and 'repel' representations of different images, the SimCLR-output matrix should match to [I|O|I|O], with I = identity-matrix and O = zero-matrix.
  For this purpose, a custom Keras-layer is defined: [SoftmaxCosineSim](SoftmaxCosineSim.py) (see [notebook](0_illustration_SoftmaxCosineSim.ipynb) for intuitive toy example).
  4. A simple Keras cross_entropy-loss can be used to evaluate the difference between the SimCLR-output and [I|O|I|O].
  5. As such, the SimCLR_model can be trained and simultaneously the feature encoding improves.

Difference with official [implementation](https://github.com/google-research/simclr):

  * Swish activation instead of relu in projection head
  * As only 1 device is used, no global batch normalization
  * Only colour distortion used with reduced color_jitter strength of 0.5 instead of 1.0. Possible to activate other distortions in [DataGeneratorSimCLR](DataGeneratorSimCLR.py).
  * Adam optimizer instead of Lars, no warmup nor cosine decay on learning rate, reduction on plateau instead.

# Experiments

SimCLR has been used as a self-supervised learning approach to improve the feature encoding quality of a pretrained VGG16-network. A SimCLR_model has been built around the base_model and consequently trained on the SimCLR-task. For this, a gradual defreeze of the base model was adopted. A clear improvement of the feature representations could be observed for the downstream classification task.

### Data: Trashnet

The [trashnet-dataset](https://github.com/garythung/trashnet) has been used.
The original dataset has been reduced to 5 classes with the following number of instances:
  * Glass: 501
  * Paper: 594
  * Cardboard: 403
  * Plastic: 482
  * Metal: 410

The original images of (512x384) have been center-cropped and reduced to a size (80x80).
Data has been split in train/val/test - 70/15/15.

Note that the similar results have been observed on a private dataset, see project context below.

### Evaluation

The feature quality is evaluated by the means of
  * A linear classifier (logistic regression) trained on the extracted features of the encoder
  * A fine-tuned classifier. 5 attempts are performed, the best classifier is kept.
  * A t-SNE visualization is made.

These evaluations are done for 3 fractions of the training data: 100%, 20%, 5%.


### Results

The table below lists the top-1 accuracy for all cases. It can be seen that SimCLR improves the classification performance for all fractions of the training set on both the linear and fine-tuned classifier.

One can consequently conclude that the feature encoding of the base_model clearly improves thanks to the SimCLR framework.

<p align="center">

|   Fraction of training set   |  Classifier   | VGG16      |  SimCLR |
|:----------:|:-------------:|:-------------:|:------:|
| 100% | Linear | 0.79 ± 0.00 | 0.82 ± 0.01
|      | Fine-tuned | 0.85 ± 0.01| 0.87 ± 0.01
| 20% | Linear | 0.70 ± 0.00| 0.81 ±0.02
|      | Fine-tuned | 0.83 ± 0.01| 0.86 ± 0.01
| 5% | Linear | 0.63 ± 0.00| 0.80 ± 0.02
|      | Fine-tuned | 0.80 ± 0.02| 0.84 ± 0.03


<small>Since the results change slightly because of the stochastic nature of the optimization procedure of both the SimCLR_model and the fine-tuned classifier, the average and standard deviation over 10 runs are presented in the table above.



# Project Context

This repository is part of a joined research project of [KU Leuven](https://www.kuleuven.be/kuleuven/), [Sagacify](https://sagacify.com/) and [BESIX](https://www.besix.com/en) on the topic of automatic monitoring of waste containers on construction sites. For this purpose, data has been collected during a period of 5 months. Similar results where achieved on this dataset. See below for an illustration of the type of data.
If you would be interested in the details of this research, please feel free to reach out.

<p align="center">

  <img src=/img/container1.png alt="alt text" width="300"/>
  <img src=/img/container2.png alt="alt text" width="300"/>
  <br>
  <e> Fig.3 - Illustration of ContAIner output </e>
</p>
