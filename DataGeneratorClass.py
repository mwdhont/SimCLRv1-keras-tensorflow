import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.utils import data_utils


class DataGeneratorClass(data_utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=16,
        subset="train",
        shuffle=True,
        preprocess=None,
        info={},
        max_width=80,
        max_height=80,
        num_classes=5,
        VGG=True,
        augmentation=None,
    ):
        super().__init__()
        self.df = df
        self.indexes = np.asarray(self.df.index)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.max_width = max_width
        self.max_height = max_height
        self.num_classes = num_classes
        self.VGG = VGG
        self.augmentation = augmentation
        self.datagen = self.datagen()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def datagen(self):
        return ImageDataGenerator()

    def __getitem__(self, index):
        X = np.empty(
            (self.batch_size, self.max_height, self.max_width, 3),
            dtype=np.float32,
        )
        y = np.empty((self.batch_size, self.num_classes), dtype=np.float32,)

        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_data = self.df.iloc[indexes]
        for i, row in enumerate(batch_data.iterrows()):
            filename = row[1]["filename"]
            self.info[index * self.batch_size + i] = filename
            img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
            if self.VGG:
                X[i,] = preprocess_input(np.asarray(img))
            else:
                # Other preprocessing to be implemented in correspondence with chosen model
                X[i,] = img

            if self.subset == "train":
                y[i,] = row[1]["class_one_hot"]

        if self.preprocess is not None:
            X = self.preprocess(X)

        # [None] is used to silence warning
        # https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to
        return X, y, [None]
