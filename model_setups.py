from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import BatchNormalization,LeakyReLU,Dense,Dropout
from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Flatten,ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow import keras

import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.keras import layers, models


# Now, create spectrogram datasets from the audio datasets
def make_spec_ds(ds):
    return ds.map(map_func=lambda audio,label: (get_spectrogram(audio), label), num_parallel_calls=tf.data.AUTOTUNE)



# Create a utility function for converting waveforms to spectrograms
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
        

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def kws_ieeespl_model(train_spectrogram_ds, input_shape, num_labels):

    """
    Architecture used in [1] for performing keyword spotting using the
    multitaper mel-spectrogram features. The architecture below is a 
    simplified version of the architecture of [2] and [3], which has
    been simplified to have less than 250k parameters.

    [1] https://ieeexplore.ieee.org/abstract/document/9896956

    [2] T. N. Sainath and C. Parada, “Convolutional neural networks for 
        small footprint keyword spotting,” in Proc. Int. Speech Commun. 
        Assoc. (INTERSPEECH), Dresden, Germany, Sep. 2015, pp. 1478–1482.

    [3] https://www.tensorflow.org/tutorials/audio/simple_audio

    """
    
    # Normalization layer
    norm_layer = layers.Normalization()
    norm_layer.adapt(train_spectrogram_ds.map(lambda spec, label: spec))

    # Define the model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_labels)
    ])

    return model



def mey_vibration_model(n_dimension_0, 
                        n_conv_layers, 
                        filter_size, 
                        use_batch_normalization, 
                        n_dense_units,
                        dropout_rate):
    

    """
    Architecture proposed by Oliver Mey et al. to classify vibration signals
    from motor's rotating shaft.

    https://github.com/deepinsights-analytica/ieee-etfa2020-paper
    https://ieeexplore.ieee.org/document/9212000

    """

    X_in = Input(shape=(n_dimension_0,1))
    x = X_in
    for j in range(n_conv_layers):
        print(j)
        x = Conv1D(filters=(j+1)*10,
                   kernel_size=filter_size,
                   strides=1,
                   activation='linear',
                   kernel_initializer='he_uniform')(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = MaxPooling1D(pool_size=5, strides=2)(x)
    x = Flatten()(x)
    x = Dense(units = n_dense_units, activation='linear')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x)
    X_out = Dense(units = 1, activation = 'sigmoid')(x)
    classifier = Model(X_in, X_out)

    return classifier


def keras_eeg_model(num_classes):

    """
    Architecture employed by Suvaditya Mukherjee for action identification
    in EEG signals, as shown in the keras tutorial link below.
 
    https://keras.io/examples/timeseries/eeg_signal_classification/

    """

    input_layer = keras.Input(shape=(512, 1))

    x = layers.Conv1D(
        filters=32, kernel_size=3, strides=2, activation="relu", padding="same"
    )(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=128, kernel_size=5, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=256, kernel_size=5, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=512, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=1024,
        kernel_size=7,
        strides=2,
        activation="relu",
        padding="same",
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        2048, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        1024, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer)




def keras_cnn_tf_from_scratch(input_shape, num_classes):
    """
    CNN architecture Keras tutorial for time series classification from
    
    https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    
    """
   

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
    


# 


class magar_faultnet(nn.Module):

    """
    Neural network architecture for time series classification adapted from
        
    CNN (FautNet) adapted from
    https://ieeexplore.ieee.org/document/9345676

    Codes adapted from
    https://github.com/BaratiLab/FaultNet

    """


    def __init__(self):
        super(magar_faultnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4, stride=1)
        self.mp2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.fc1= nn.Linear(2304,256)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = x.view(in_size,-1)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x)
       
        return F.log_softmax(x, dim=1)