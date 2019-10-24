import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0' # necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations
np.random.seed(42)  #  starting Numpy generated random numbers in a well-defined initial state
rn.seed(12345) # starting core Python generated random numbers in a well-defined state
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) # Force TensorFlow to use single thread.
from keras import layers
from keras.layers.core import Dense
from keras.layers import Conv1D,Conv2D
from keras.optimizers import Adam
from keras.losses import squared_hinge
from keras.losses import categorical_crossentropy
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LSTM, GRU
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers.core import Reshape
from keras.initializers import glorot_normal
from keras.regularizers import l2
import keras.backend as K
from keras.layers.merge import concatenate
from keras.layers import Flatten, Add
from keras.models import Model


from tcn import TCN
import math
def build_model(batchSize, numTimeSteps=98, numTargets=12, numFeats=10):
  DropoutProb = 0.2  # Dropout probability [%]
  FIRST_LAYER_NEURONS=16
  NEXT_LAYERS_NEURONS=48
  L2_lambda = 0.0008

  # Training parameters
  adamLearningRate = 0.001
  adamBeta1 = 0.9
  adamBeta2 = 0.999
  adamEpsilon = 1e-08
  adamDecay = 0.0

  # Make Network
  i = Input(batch_shape=(batchSize, numTimeSteps, numFeats))


  o = TCN(nb_filters=FIRST_LAYER_NEURONS,
          kernel_size=3,
          nb_stacks=1,
          dilations=[1],
          padding='causal',
          use_skip_connections=False,
          dropout_rate=DropoutProb,
          activation='linear',
          return_sequences=True,
          use_batch_norm=True,
          name='tcn_0')(i)

  o = TCN(nb_filters=NEXT_LAYERS_NEURONS,
          kernel_size=3,
          nb_stacks=1,
          dilations=[2,4,8],
          padding='causal',
          use_skip_connections=True,
          dropout_rate=DropoutProb,
          activation='relu',
          return_sequences=True,
          use_batch_norm=True,
          name='tcn_1')(o)


  o = layers.AveragePooling1D(pool_size=(4))(o)


  o = Flatten()(o)
  o= Dense(numTargets)(o)
  o = Activation('softmax', name='softmax')((o))
  
  model = Model(inputs=[i], outputs= [o])


     
  # Define optimizer
  adam = Adam(lr=adamLearningRate, beta_1=adamBeta1, beta_2=adamBeta2, epsilon=adamEpsilon, decay=adamDecay)
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  print(model.count_params())
  print(model.summary())

  #K.clear_session()    
  return model