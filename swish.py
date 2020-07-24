from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation

from tensorflow.keras.utils import get_custom_objects


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = "swish"


def swish(x):
    return K.sigmoid(x) * x


get_custom_objects().update({"swish": Swish(swish)})
