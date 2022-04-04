import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# from ops import add_features

# variables and dimensions
numSeqs = 0
numDims = 1
numTimesteps = 0
inputPattSize = 2  # x, y and pen-up

path = '../data/Unipen/train_r01_v07/data/1a/'
data_path = '../data/Unipen/train_r01_v07/include/'
save_path = '../data/Unipen/newUnipen/'
vocabulary = '0123456789'


def is_data_point(inputString):
    point_list = inputString.split(' ')
    if len(point_list) == 2:
        for point in point_list:
            if not all(char.isdigit() or char == ' ' for char in point):
                return False
        return True
    else:
        return False


def get_data_file(file_content):
    for line in file_content:
        line = line.replace('\n', '')
        if line.find('.INCLUDE') != -1 and line.find('.dat') != -1:
            return open(os.path.join(data_path, line.replace('.INCLUDE ', '')), "r")

def LoadUnipen():
    # Process all active samples in the sets
    X = []
    y = []
    pen_up = []
    for writer in os.listdir(path):
        writer_path = os.path.join(path, writer)
        for root, dirs, files in os.walk(writer_path):
            for session in files:
                #print("processing file: ", root + "/" + session)
                session_file_lines = open(os.path.join(root, session), "r").readlines()
                data_file = get_data_file(session_file_lines)
                strokes = []
                current_stroke = []
                for line in data_file.readlines():
                    line = line.replace('\n', '')
                    if line.find('.PEN_UP') != -1:
                        strokes.append(current_stroke)
                        current_stroke = []
                    elif is_data_point(line):
                        stroke_point = line.split(' ')
                        current_stroke.append([float(stroke_point[0]), float(stroke_point[1])])

                for line in session_file_lines:
                    if line.find('.SEGMENT CHARACTER') != -1:
                        if line.find(' ? ') != -1:
                            index_and_label = line.split('?')
                        else:
                            index_and_label = line.split('OK')

                        char_idx = index_and_label[0]
                        char_idx = char_idx.replace('.SEGMENT CHARACTER ', '').replace(' ', '')
                        char_idx = char_idx.split('-')
                        char_idx = list(map(int, char_idx))
                        if len(char_idx) > 1:
                            char_idx = list(range(char_idx[0], char_idx[-1]+1))

                        label = index_and_label[1].replace(' ', '').replace('"', '').replace('\n', '')

                        sequence = []
                        for idx in char_idx:
                            sequence.extend(strokes[idx])

                        # add features
                        sequence = np.asarray(sequence)
                        sequence = sequence
                        onehot_label = np.zeros(len(vocabulary))
                        if label.upper() in vocabulary:
                            X.append(sequence)
                            onehot_label[vocabulary.index(label.upper())] = 1.
                            y.append(onehot_label)
                        # elif len(sequence) > 0 and len(sequence) <= 50:
                        #     sequence = np.append(sequence, [[0, 0]] * (50 - len(sequence)), 0)
                        #     onehot_label = np.zeros(len(vocabulary))
                        #     if label.upper() in vocabulary:
                        #         X.append(sequence)
                        #         onehot_label[vocabulary.index(label.upper())] = 1.
                        #         y.append(onehot_label)
                        # sequence = add_features(sequence)


    # normalize data and add pen_up column
    X = np.asarray(X)
    print(X.shape)
    # Save to file
    return(X, y)
