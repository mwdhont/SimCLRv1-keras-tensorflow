import numpy as np
import cv2 as cv

import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.utils import data_utils

from SimCLR_data_util import *
import random


class DataGeneratorSimCLR(data_utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=16,
        subset="train",
        shuffle=True,
        info={},
        width=80,
        height=80,
        VGG=False,
    ):
        super().__init__()
        self.df = df
        self.indexes = np.asarray(self.df.index)
        self.batch_size = batch_size
        self.subset = subset
        self.shuffle = shuffle
        self.info = info
        self.width = width
        self.height = height
        self.VGG = VGG
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty(
            (2 * self.batch_size, 1, self.height, self.width, 3),
            dtype=np.float32,
        )

        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_data = self.df.iloc[indexes]

        # Add shuffle in order to avoid network recalling fixed order
        shuffle_a = np.arange(self.batch_size)
        shuffle_b = np.arange(self.batch_size)

        if self.subset == "train":
            random.shuffle(shuffle_a)
            random.shuffle(shuffle_b)
        if self.subset == "val":
            # Exclude randomness for evaluation
            random.seed(42)
            random.shuffle(shuffle_a)
            random.shuffle(shuffle_b)

        labels_ab_aa = np.zeros((self.batch_size, 2 * self.batch_size))
        labels_ba_bb = np.zeros((self.batch_size, 2 * self.batch_size))

        for i, row in enumerate(batch_data.iterrows()):
            filename = row[1]["filename"]
            self.info[index * self.batch_size + i] = filename
            img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)

            img = tf.convert_to_tensor(
                np.asarray((img / 255)).astype("float32")
            )

            img_T1 = preprocess_for_train(
                img,
                self.height,
                self.width,
                color_distort=True,
                crop=False,
                flip=False,
                blur=False,
            )
            img_T2 = preprocess_for_train(
                img,
                self.height,
                self.width,
                color_distort=True,
                crop=False,
                flip=False,
                blur=False,
            )

            if self.VGG:
                img_T1 = tf.dtypes.cast(img_T1 * 255, tf.int32)
                img_T2 = tf.dtypes.cast(img_T2 * 255, tf.int32)
                img_T1 = preprocess_input(np.asarray(img_T1))
                img_T2 = preprocess_input(np.asarray(img_T2))

            # T1-images between 0 -> batch_size - 1
            X[shuffle_a[i]] = img_T1
            # T2-images between batch_size -> 2*batch_size - 1
            X[self.batch_size + shuffle_b[i]] = img_T2

            # label ab
            labels_ab_aa[shuffle_a[i], shuffle_b[i]] = 1
            # label ba
            labels_ba_bb[shuffle_b[i], shuffle_a[i]] = 1

        y = tf.concat([labels_ab_aa, labels_ba_bb], 1)

        # [None] is used to silence warning
        # https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to
        return list(X), y, [None]
