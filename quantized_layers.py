# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.layers import InputSpec, Dense, Conv2D
from keras import constraints
from keras import initializers
import tensorflow as tf
from .util_layers import Clip
from quantization_functions.quantized_ops import quantize_round, quantize_floor

# Quantized Fully Connected Layer
class QuantizedDense(Dense):
    def __init__(self, units, H=1., nb=16, kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.nb = nb
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        super(QuantizedDense, self).__init__(units, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))
            
        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
	
        quantized_inputs=quantize_round(inputs, nb=self.nb)
        quantized_kernel = quantize_round(self.kernel, nb=self.nb)
        quantized_bias=quantize_round(self.bias, nb=self.nb)
        output = K.dot(quantized_inputs, quantized_kernel)
        if self.use_bias:
            output = K.bias_add(output, quantized_bias)
            
        output=quantize_floor(output, nb=self.nb)
        #output =tf.Print(output,[output], message="output without activation")    
        if self.activation is not None:
            output = self.activation(output)
        
        #output=quantize(output, nb=self.nb)
	
        return output
        
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


