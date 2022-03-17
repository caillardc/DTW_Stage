from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
from CLayer.CLayer import *
from tslearn.datasets import UCR_UEA_datasets
from tensorflow.keras.utils import to_categorical
import yaml

with open(r'config.yaml') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(config["data_name"])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

config_dwa = config["model_dwa"]

def create_model():
    input = Input(X_train.shape[1:])
    if config_dwa["model"]["n_conv_layer"] != 1:
        x = DWA_CNN(config_dwa["conv1d"])(input)
        for i in range(config_dwa["n_conv_layer"]):
            x = DWA_CNN(config_dwa["conv1d"])(x)
    else:
        x = DWA_CNN(**config_dwa["conv1d"])(input)
    x = Flatten()(x)
    output_layer = Dense(3, activation='softmax')(x)
    return Model(input,output_layer)

model = create_model()
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=[X_test, y_test])

