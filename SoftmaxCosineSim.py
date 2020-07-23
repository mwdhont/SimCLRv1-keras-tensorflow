# ==============================================================================
# Code modified from NT-XENT-loss:
# https://github.com/google-research/simclr/blob/master/objective.py
# ==============================================================================
# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax

from swish import swish


class SoftmaxCosineSim(keras.layers.Layer):
    """Custom Keras layer: takes all z-projections as input and calculates
    output matrix which needs to match to [I|O|I|O], where
            I = Unity matrix of size (batch_size x batch_size)
            O = Zero matrix of size (batch_size x batch_size)
    """

    def __init__(self, batch_size, feat_dim, **kwargs):
        super(SoftmaxCosineSim, self).__init__()
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.units = (batch_size, 4 * feat_dim)
        self.input_dim = [(None, feat_dim)] * (batch_size * 2)
        self.temperature = 0.1
        self.LARGE_NUM = 1e9

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "batch_size": self.batch_size,
                "feat_dim": self.feat_dim,
                "units": self.units,
                "input_dim": self.input_dim,
                "temperature": self.temperature,
                "LARGE_NUM": self.LARGE_NUM,
            }
        )
        return config

    def call(self, inputs):
        z1 = []
        z2 = []

        for index in range(self.batch_size):
            # 0-index assumes that batch_size in generator is equal to 1
            z1.append(tf.math.l2_normalize(inputs[index][0], -1))
            z2.append(
                tf.math.l2_normalize(inputs[self.batch_size + index][0], -1)
            )

        # Gather hidden1/hidden2 across replicas and create local labels.
        z1_large = z1
        z2_large = z2

        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

        # Products of vectors of same side of network (z_i), count as negative examples
        # Values on the diagonal are put equal to a very small value
        # -> exclude product between 2 identical values, no added value
        logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM

        logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM

        # Similarity between two transformation sides of the network (z_i and z_j)
        # -> diagonal should be as close as possible to 1
        logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / self.temperature

        part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
        part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
        output = tf.concat([part1, part2], 1)

        return output
