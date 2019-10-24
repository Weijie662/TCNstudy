# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from keras import backend as K

from keras.layers import InputSpec, Layer, Dense, Conv2D
from keras import constraints
from keras import initializers
from keras import activations
from keras import regularizers

from keras.utils.generic_utils import has_arg


# Legacy support.
from keras.legacy import interfaces

from quantization_functions.quantized_ops import clip_through

class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None, clip_through=False):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value
        self.clip_through = False

    def __call__(self, p):
        if self.clip_through:
            return clip_through(p, self.min_value, self.max_value)
        else:
            return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"name": self.__call__.__name__,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "clip_through": self.clip_through}
