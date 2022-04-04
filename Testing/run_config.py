from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import os
from CLayer.CLayer import *
from tensorflow.keras.utils import to_categorical
import yaml
import copy
import sys
import pickle
from CLayer.loadData import load_dataset
from tensorboard.plugins.hparams import api as hp


def create_model(config, X_train, y_train):
    input = Input(X_train.shape[1:])

    #CONV LAYERS

    if config["n_conv_layer"] != 1:
        x = eval(config["conv_layer_type"])(**config["conv1d_0"])(input)
        if config["batch_norm"]:
            x = BatchNormalization()(x)
        for i in range(1, config["n_conv_layer"]):
            x = eval(config["conv_layer_type"])(**config["conv1d_"+str(i)])(x)
            if config["batch_norm"]:
                x = BatchNormalization()(x)
    else:
        x = eval(config["conv_layer_type"])(**config["conv1d"])(input)
        if config["batch_norm"]:
            x = BatchNormalization()(x)
    x = Flatten()(x)

    #DENSE LAYERS
    if config["n_dense_layer"] != 1:
        x = Dense(**config["dense_layer_0"])(x)
        for i in range(1, config["n_dense_layer"]):
            x = Dense(**config["dense_layer_"+str(i)])(x)
    else:
        x = Dense(**config["conv1d"])(input)
    output_layer = Dense(y_train.shape[1], activation='softmax')(x)
    return Model(input,output_layer)


def run(logdir, config, dataset, hparams, session_num):
    X_test, y_test, X_train, y_train = dataset
    model = create_model(config, X_train, y_train)
    if config["optimizer"] == "SGD":
        optimizer = eval(config["optimizer"])(learning_rate=config["learning_rate"], decay=config["decay_rate"])
    else:
        optimizer = eval(config["optimizer"])(learning_rate=config["learning_rate"])
    model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=config["batch_size"] ,epochs=config["epochs"], validation_data=[X_test, y_test],
              callbacks=[
                  tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                  hp.KerasCallback(logdir, hparams)
              ])
    model.save_weights("{}/weights_{:.4f}_{}.h5".format(config["experiment_name"], hist.history['val_accuracy'][-1], session_num))
    model.save("{}/model_{:.4f}_{}.h5".format(config["experiment_name"], hist.history['val_accuracy'][-1], session_num))


if __name__ == "__main__":

    try: sys.argv[1]
    except IndexError:
        with open(r'iwana_config.yml') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        if not os.path.exists(config["experiment_name"]):
            os.makedirs(config["experiment_name"])
        with open(os.path.join(os.getcwd(), config["experiment_name"] + '/' + 'iwana_config.yml'), 'w') as file:
            documents = yaml.dump(config, file)
    else:
        with open(sys.argv[1]) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        if not os.path.exists(config["experiment_name"]):
            os.makedirs(config["experiment_name"])
        with open(os.getcwd() + '/' + config["experiment_name"] + '/' + sys.argv[1], 'w') as file:
            documents = yaml.dump(config, file)


    dataset = (load_dataset(config["data_type"], config["data_name"]))
    session_num = 0
    for conv_type in config["conv_layer_type"]:
        for batch_norm in config["batch_norm"]:
            for decay in config["decay_rate"]:
                for learning_rate in config["learning_rate"]:
                    for optimizer in config["optimizer"]:

                        config_i = copy.deepcopy(config)
                        config_i["conv_layer_type"] = conv_type
                        config_i["batch_norm"] = batch_norm
                        config_i["decay_rate"] = decay
                        config_i["learning_rate"] = learning_rate
                        config_i["optimizer"] = optimizer

                        hparams = {
                            'HP_conv_layer_type': conv_type,
                            'HP_batch_norm': batch_norm,
                            'HP_decay_rates': decay,
                            'HP_learning_rate': learning_rate,
                            'HP_optimizer': optimizer,
                        }

                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h : hparams[h] for h in hparams})
                        run(config_i["experiment_name"] + '/logs/hparam_tuning/' + run_name, config_i, dataset, hparams, session_num)
                        session_num += 1


