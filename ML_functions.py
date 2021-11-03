import numpy as np

import tensorflow as tf
from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)

import librosa
import librosa.display

import pickle

#Defining key variables
sampling_rate = 44100
audio_duration = 2.5
n_mfcc = 30
class_mapping = [
                   'female_angry', 'female_disgust', 'female_fear', 'female_happy',
                   'female_neutral', 'female_sad', 'female_surprise', 'male_angry',
                   'male_disgust', 'male_fear', 'male_happy', 'male_neutral',
                   'male_sad', 'male_surprise'
                ]


def prepare_data(audio_link,n):
    X = np.empty(shape = (1,n,216,1))
    input_length = sampling_rate * audio_duration

    #Loading the audio file
    data , _ = librosa.load(audio_link,
                            sr = sampling_rate,
                            res_type = "kaiser_fast",
                            duration = 2.5,
                            offset = 0.5)
    #Random offset /Padding
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
    MFCC = librosa.feature.mfcc(data , sr = sampling_rate, n_mfcc = n_mfcc)
    MFCC = np.expand_dims(MFCC,axis = -1)
    print(MFCC.shape)
    return MFCC


def get_2d_conv_model(n):
    ''' Create a standard deep 2D convolutional neural network'''
    nclass = 14
    inp = Input(shape=(n, 216, 1))
    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)

    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def load_mean_std():
    with open('ml_model/mean.pkl','rb') as f:
        mean = pickle.load(f)
    with open('ml_model/std.pkl','rb') as f:
        std = pickle.load(f)
    return mean,std




if __name__ == '__main__':
    audio_link = "ml_model/03-01-01-01-01-02-01.wav"
    data = prepare_data(audio_link, n = n_mfcc)
    mean,std = load_mean_std()
    data = (data - mean)/std
    print(data.shape)
    model = tf.keras.models.load_model('ml_model/my_model')
    probs = model.predict(np.expand_dims(data,axis = 0))
    print(f"Prediction class is {class_mapping[np.argmax(probs)]}")


