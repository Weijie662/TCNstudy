###### Training for Google Speech Command Dataset using LSTMs ########################

This script allows to train an LSTM model (floating point or quantized) for voice command recognition based on the Google Speech Command Dataset. The task is classyfing 10 voice commands plus unknow label and silence label. The mentioned commands are 'yes','no','up','down','left','right','on','off','stop' and 'go'.
The quantization is embedded in the definition of the customized classes developed for keras. 

### USE ##############################################################

To run the training scripts

python train.py

The script builds a model based on the declaration done in 'models/BuildLSTMClassifier.py'. For training purposes, a generator is used to produce a different set of the training data for each epoch. The data is augmented through random time shifting, noise addition as well as the selection of different samples for the unknwon target.

### FOLDERS ####################################################################

models:
This folder contains the definition of the neural architecture to train. 

Quantization_functions:
Files declaring the quantized operations (quantized_ops.py) as well as the piecewise linear approximation of the nonlinear functions (piecewise_linear_approximation.py).

layers:
Files containing the classes correspondent to the quantized fully connected layer (quantized_layers.py) and LSTM layers (quantized_layers_recurrent.py). These files embed the quantized operations assumed in the hardware platform into the training process.

pretrained_models:
Pretrained models saved from previously training sessions. fp means Floating Point, fxp means Fixed Point, hl refers to hidden layers and n to number of neurons.

utils:
The file load_data_google_dataset.py loads a subset of 22,000 examples when it is called. Different parameters could be tuned as the training, validation and test partitions, window size for mfcc extraction, etc.


weights:
The models are saved through hdf5 format. Every version of the weights is saved to model.hdf5. For quantization training, a pretrained floating point model can be used to accelerate training convergence.

logs:
Logs of the execution.



