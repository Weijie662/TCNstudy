import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
import keras.backend as K
from BuildTCNClassifier import build_model
from load_data_google_dataset import gsc_dataset
import numpy as np
from keras.backend.tensorflow_backend import set_session

# Session configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=config)
set_session(sess)


#Parameters
batch_size=256
numTimeSteps=61
numFeats=10
numTargets=12
epochs=100000
cycles_per_subdataset=1 # Number of times the subset is passed through training
batch_size_train=256 # Size of each subset taken for training
dataset='GOOGLE-DATASET'
name='lstm'
progress_logging=1
tensorboard_name = '{}_{}.hdf5'.format(dataset,name)


# ## Construct the network
print('Construct the Network\n')
model = build_model(batchSize=batch_size, numTimeSteps=numTimeSteps, numTargets=numTargets, numFeats=numFeats)
print('setting up the network and creating callbacks\n')
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights/model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
# tensorboard = TensorBoard(log_dir='./logs/fxp_' + str(tensorboard_name), histogram_freq=0, write_graph=True, write_images=False)


## Scheduler
def scheduler(epoch):
    return K.get_value(model.optimizer.lr)
lr_decay = LearningRateScheduler(scheduler)


# Loading validation and test sets of Dataset
dt=gsc_dataset(numFeats=numFeats)
tr, v, tst, tr_y,v_y,tst_y = dt.load_dataset_google_dataset(batch_size_train=0)





# Generator() produces the input data as well as the correspondent labels
def generator():
 while True:
  TR, v, tst, TR_Y,v_y,tst_y = dt.load_dataset_google_dataset(batch_size_train=batch_size, batch_size_val=0, batch_size_test=0)
  yield TR, TR_Y



#model.load_weights('weights/model.hdf5', by_name=True) # Optional loading of model
log_dir="/tmp/tensorboard_logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
model.fit_generator(generator(), 
                            steps_per_epoch=1,
                            epochs=epochs,
                            verbose=progress_logging,
                            # callbacks = [tensorboard_callback, checkpoint,lr_decay],
                            initial_epoch= 0,
                            validation_data = (v,v_y) )

# Test accuracy
a,accuracy_tst=model.evaluate(x=tst, y=tst_y, batch_size=batch_size, verbose=1)
print("Test Accuracy")
print(accuracy_tst)







