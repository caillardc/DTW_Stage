import os
import numpy as np
from PIL import Image as im
import csv
import pickle
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
from random import shuffle

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = int(np.amax(labels_dense) + 1)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, images, labels, one_hot=False):
        #images_2 is series
        assert images.shape[0] == labels.shape[0], ('images_1.shape: %s labels_1.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # assert images.shape[3] == 1
        #images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
        # Convert from [0, 255] -> [-1.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 127.5) - 1.
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images


    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            print("epoch " + str(self._epochs_completed))
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def load_data(data_file, label_file, image_shape, onehot):
    images = np.genfromtxt(data_file, delimiter=' ')
    labels = np.genfromtxt(label_file, usecols=(1), delimiter=' ')
    if onehot:
        labels = dense_to_one_hot(labels.astype(int))

    return images, labels

def load_data_from_file(data_file, label_file, image_shape, onehot):
    labelsall = np.genfromtxt(label_file, delimiter=' ', dtype=None)
    labelsshape = np.shape(labelsall)

    images = np.zeros((labelsshape[0], image_shape[0], image_shape[1], image_shape[2]))
    labels = np.zeros((labelsshape[0]))
    count = 0
    for line in labelsall:
        labels[count] = line[1]
        imagefile = im.open(data_file + line[0].decode("utf-8"))
        imagefile = imagefile.convert('RGB')
        images[count] = np.array(imagefile)
        imagefile.close()
        if count % 1000 == 0:
            print(count)
        count += 1
    if onehot:
        labels = dense_to_one_hot(labels.astype(int))

    return images, labels

def read_data_sets(train_file, train_label, shape, test_file="", test_label="", test_ratio=0.1, validation_ratio=0.0, pickle=True, boring=False, onehot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()

    if (pickle):
        train_images, train_labels = load_data_from_pickle(train_file, train_label, shape)
        if test_file:
            test_images, test_labels = load_data_from_pickle(test_file, test_label, shape)
        else:
            test_size = int(test_ratio * float(train_labels.shape[0]))
            test_images = train_images[:test_size]
            test_labels = train_labels[:test_size]
            train_images = train_images[test_size:]
            train_labels = train_labels[test_size:]
    elif(boring):
        train_images, train_labels = load_data_from_file(train_file, train_label, shape, onehot)
        if test_file:
            test_images, test_labels = load_data_from_file(test_file, test_label, shape, onehot)
        else:
            test_size = int(test_ratio * float(train_labels.shape[0]))
            test_images = train_images[:test_size]
            test_labels = train_labels[:test_size]
            train_images = train_images[test_size:]
            train_labels = train_labels[test_size:]
    else:
        train_images, train_labels = load_data(train_file, train_label, shape, onehot)
        if test_file:
            test_images, test_labels = load_data(test_file, test_label, shape, onehot)
        else:
            test_size = int(test_ratio * float(train_labels.shape[0]))
            test_images = train_images[:test_size]
            test_labels = train_labels[:test_size]
            train_images = train_images[test_size:]
            train_labels = train_labels[test_size:]

    validation_size = int(validation_ratio * float(train_labels.shape[0]))
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]

    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)

    print("data loaded")
    return data_sets


def load_unipen(dataset="1a"):

    TRAINING_FILE = "../../data/Unipen/{}-re-data.txt".format(dataset)
    TRAINING_LABEL = "../../data/Unipen/{}-re-labels.txt".format(dataset)

    CONV_OUTPUT_SHAPE = 7 #50 25 13 7
    MPOOL_SHAPE = 2
    IMAGE_SHAPE = (50, 2)


    data_sets = read_data_sets(TRAINING_FILE, TRAINING_LABEL, IMAGE_SHAPE,
                                           validation_ratio=0.3, pickle=False, boring=False, onehot=True)

    train_data = (data_sets.train.images.reshape((-1, 50, 2)) + 1. ) * (127.5 / 127.)  # Returns np.array
    train_labels = data_sets.train.labels
    eval_data = (data_sets.test.images.reshape((-1, 50, 2)) + 1. ) * (127.5 / 127.)  # Returns np.array
    eval_labels = np.asarray(data_sets.test.labels, dtype=np.int32)
    scaler = MinMaxScaler((-0.5, 0.5))
    train_data = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    eval_data = scaler.transform(eval_data.reshape(-1, eval_data.shape[-1])).reshape(eval_data.shape)
    return train_data, train_labels, eval_data, eval_labels


def get_filepaths(mainfolder):
    """
    Searches a folder for all unique files and compile a dictionary of their paths.
    Parameters
    --------------
    mainfolder: the filepath for the folder containing the data
    Returns
    --------------
    training_filepaths: file paths to be used for training
    testing_filepaths:  file paths to be used for testing
    """
    training_filepaths = {}
    testing_filepaths  = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            filenames = os.listdir(fpath)
            for filename in filenames[:int(round(0.85*len(filenames)))]:
                fullpath = fpath + "/" + filename
                training_filepaths[fullpath] = folder
            for filename1 in filenames[int(round(0.85*len(filenames))):]:
                fullpath1 = fpath + "/" + filename1
                testing_filepaths[fullpath1] = folder
    return training_filepaths, testing_filepaths

def get_labels(mainfolder):
    """ Creates a dictionary of labels for each unique type of motion """
    labels = {}
    label = 0
    for folder in os.listdir(mainfolder):
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            labels[folder] = label
            label += 1
    return labels

def get_data(fp, labels, folders):
    """
    Creates a dataframe for the data in the filepath and creates a one-hot
    encoding of the file's label
    """
    data = pd.read_csv(filepath_or_buffer=fp, sep=' ', names = ["X", "Y", "Z"])
    one_hot = np.zeros(7)
    file_dir = folders[fp]
    label = labels[file_dir]
    one_hot[label] = 1
    return data, one_hot, label

# Normalizes the data by removing the mean

def subtract_mean(input_data):
    # Subtract the mean along each column
    centered_data = input_data - input_data.mean()
    return centered_data


def norm_data(data):
    """
    Normalizes the data.
    For normalizing each entry, y = (x - min)/(max - min)
    """
    c_data = subtract_mean(data)
    mms = MinMaxScaler()
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def vectorize(normed):
    """
    Uses a sliding window to create a list of (randomly-ordered) 300-timestep
    sublists for each feature.
    """
    sequences = [normed[i:i+150] for i in range(len(normed)-150)]
    shuffle(sequences)
    sequences = np.array(sequences)
    return sequences

def build_inputs(files_list, accel_labels, file_label_dict):
    X_seq    = []
    y_seq    = []
    labels = []
    for path in files_list:
        data, target, target_label = get_data(path, accel_labels, file_label_dict)
        # if len(data) < 200:
        #     data = pd.concat((data, pd.DataFrame([[0, 0, 0]]*(200-len(data)), columns = ["X", "Y", "Z"])))
        # input_list = np.array(data[0:200])
        input_list = np.array(data)
        X_seq.append(input_list)
        y_seq.append(list(target))
        labels.append(target_label)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    return X_, y_, labels

def load_UCI_ADL():
    mainpath = "../data/HMP_Dataset"
    activity_labels                  = get_labels(mainpath)
    training_dict, testing_dict      = get_filepaths(mainpath)
    training_files                   = list(training_dict.keys())
    testing_files                    = list(testing_dict.keys())
    X_train, y_train, train_labels = build_inputs(
        training_files,
        activity_labels,
        training_dict)
    X_test, y_test, test_labels = build_inputs(
        testing_files,
        activity_labels,
        testing_dict)
    shuffle = np.random.permutation(len(X_train))
    X_train = X_train[shuffle]
    y_train = y_train[shuffle]
    return X_test, y_test, X_train, y_train

def load_dataset(datatype, data_name):
    if datatype == "Unipen":
        TRAINING_FILE = "../../data/Unipen/NewUnipen/unipen_X.pkl"
        TRAINING_LABEL = "../../data/Unipen/NewUnipen/unipen_y.pkl"
        x = np.array(pickle.load(open(TRAINING_FILE, "rb")))
        x = x.reshape(len(x), 50, 2)
        y = np.array(pickle.load(open(TRAINING_LABEL, "rb")))
        X_train = x[:int(len(x)*0.9)]
        X_test = x[int(len(x)*0.9):]
        y_train = y[:int(len(y)*0.9)]
        y_test = y[int(len(y)*0.9):]
        return X_test, y_test, X_train, y_train
    elif datatype == "UCI":
        if data_name=="ADL":
            return load_UCI_ADL()
    elif data_type == "UCR_UEA_datasets":
        X_test, y_test, X_train, y_train = UCR_UEA_datasets().load_dataset(data_name)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return X_test, y_test, X_train, y_train

