import numpy as np
import tensorflow as tf
import input_data
import models_gd
from keras.backend.tensorflow_backend import set_session
class gsc_dataset:
      def __init__(self,numFeats=10):
            self.sess = tf.Session(graph=tf.get_default_graph())
            data_url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
            data_dir = '/tmp/speech_dataset/'
            wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
            self.sample_rate = 16000
            self.background_frequency = 0.8
            self.background_volume = 0.1
            self.time_shift_ms = 100.0
            clip_duration_ms = 1000.0
            window_size_ms = 32.0
            window_stride_ms = 16.0
            dct_coefficient_count = numFeats
            unknown_percentage = 10.0
            silence_percentage = 10.0
            validation_percentage = 10.0
            testing_percentage = 10.0

            self.model_settings = models_gd.prepare_model_settings(
                  len(input_data.prepare_words_list(wanted_words.split(','))),
                  self.sample_rate, clip_duration_ms, window_size_ms,
                  window_stride_ms, dct_coefficient_count)
            self.audio_processor = input_data.AudioProcessor(
                  data_url, data_dir, silence_percentage,
                  unknown_percentage,
                  wanted_words.split(','), validation_percentage,
                  testing_percentage, self.model_settings)

      def load_dataset_google_dataset(self, batch_size_train=22016, batch_size_val=3072, batch_size_test=3072):



            time_shift_samples = int((self.time_shift_ms * self.sample_rate) / 1000)
            input_frequency_size = self.model_settings['dct_coefficient_count']
            input_time_size = self.model_settings['spectrogram_length']

            if batch_size_train!=0:
                  # Importing Training Data
                  train_fingerprints, train_ground_truth = self.audio_processor.get_data(
                  batch_size_train, 0, self.model_settings, self.background_frequency,
                  self.background_volume, time_shift_samples, 'training', self.sess)
                  # adjust tr labels for keras
                  train_ground_truth = train_ground_truth.astype(int)
                  n_values = np.max(train_ground_truth) + 1
                  train_ground_truth = np.eye(n_values)[train_ground_truth]
                  # Reshape
                  train_fingerprints = np.reshape(train_fingerprints, (-1, input_time_size, input_frequency_size))

            else:
                  train_fingerprints=[]
                  train_ground_truth=[]


            if batch_size_val!=0:
                  #Importing Validation Data
                  validation_fingerprints, validation_ground_truth = (
                  self.audio_processor.get_data(batch_size_val, 0, self.model_settings, 0.0,
                                0.0, 0, 'validation', self.sess))

                  # adjust val labels for keras
                  validation_ground_truth = validation_ground_truth.astype(int)
                  n_values = np.max(validation_ground_truth) + 1
                  validation_ground_truth = np.eye(n_values)[validation_ground_truth]
                  validation_fingerprints = np.reshape(validation_fingerprints, (-1, input_time_size, input_frequency_size))
            else:
                  validation_fingerprints=[]
                  validation_ground_truth =[]

            if batch_size_test!=0:
                  # Importing Testing Data
                  test_fingerprints, test_ground_truth = self.audio_processor.get_data(
                  batch_size_test, 0, self.model_settings, 0.0, 0.0, 0, 'testing', self.sess)

                  # adjust test labels for keras
                  test_ground_truth = test_ground_truth.astype(int)
                  n_values = np.max(test_ground_truth) + 1
                  test_ground_truth = np.eye(n_values)[test_ground_truth]
                  test_fingerprints = np.reshape(test_fingerprints, (-1, input_time_size, input_frequency_size))
            else:
                  test_fingerprints=[]
                  test_ground_truth=[]




            return train_fingerprints,validation_fingerprints,test_fingerprints,train_ground_truth,validation_ground_truth,test_ground_truth
