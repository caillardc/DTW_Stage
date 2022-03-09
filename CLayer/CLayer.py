import tensorflow as tf
import numpy as np
from dtw import *

def euclideanDistance(x, y):
    dist = tf.sqrt(tf.cast(tf.reduce_sum(tf.square(x - y)), dtype="float32"))
    return dist

def scalarDistance(x,y):
    return x * y

#Fonction qui crée la matrice DTW
@tf.function
def DTW_TF(S, S1, d=euclideanDistance):
    cost_matrix = []
    cost_matrix.append(tf.cast(tf.stack([0, *([np.inf] * S1.shape[0])]), "float32"))
    for i in range(1,S.shape[0]+1):
        sub_cost_j = [np.inf]
        for j in range(1, S1.shape[0]+1):
            dst = d(S[i-1], S1[j-1])
            mat_dt =[
                dst + sub_cost_j[j-1],
                dst + cost_matrix[i-1][j-1],
                dst + cost_matrix[i-1][j]
            ]
            sub_cost_j.append(tf.reduce_min(mat_dt))
        cost_matrix.append(tf.stack(sub_cost_j))
    return DTW_minimal_path(tf.stack(cost_matrix))


# Renvoie le chemin dtw optimal en 2 fonctions
@tf.function
def loop_function(i, j, best_path, compteur, cost_mat):
    compteur += 1
    cost_min = tf.stack([
        cost_mat[i-1, j-1],
        cost_mat[i, j-1],
        cost_mat[i-1, j]
    ])
    n_min = tf.math.argmin(cost_min)
    i, j = tf.case([
        (tf.equal(n_min,0), lambda: (i-1, j-1)),
        (tf.equal(n_min,1), lambda: (i, j-1)),
        (tf.equal(n_min,2), lambda: (i-1, j))
    ])
    best_path = best_path.write(compteur, (i-1, j-1))
    return i, j, best_path, compteur, cost_mat

@tf.function
def DTW_minimal_path(cost_mat):
    i = cost_mat.shape[0] - 1
    j = cost_mat.shape[1] - 1
    compteur = 0
    best_path = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    best_path = best_path.write(compteur, (i-1, j-1))
    cond = lambda i, j, best_path, compteur, cost_mat: tf.logical_and(tf.greater(i, 0), tf.greater(j, 0))
    i, j, best_path, compteur, cost_mat = tf.while_loop(cond, loop_function, [i, j, best_path, compteur, cost_mat])
    best_path = tf.cast(best_path.stack()[:-1][::-1], tf.int64)

    mat_allign = tf.sparse.SparseTensor(indices=best_path,
                                        values=tf.ones(tf.shape(best_path)[0], dtype=tf.dtypes.float32),
                                        dense_shape=[cost_mat.shape[0] - 1, cost_mat.shape[1] - 1])
    return mat_allign

# renvoie le chemin qui maximise la matrice DTW pour Schulman
def DTW_maximal_path(cost_mat):
    i = cost_mat.shape[0] - 1
    j = cost_mat.shape[1] - 1
    compteur = 0
    path_input = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    path_input = path_input.write(compteur, i-1)

    path_output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    path_output = path_output.write(compteur, j-1)
    while tf.greater(i, 0) and tf.greater(j, 0):
        compteur += 1
        cost_max = tf.stack([
            cost_mat[i-1, j-1],
            cost_mat[i, j-1],
            cost_mat[i-1, j]
        ])
        n_max = tf.math.argmax(cost_max)
        if tf.equal(n_max,0):
            i += -1
            j += -1
        elif tf.equal(n_max,1):
            j += -1
        elif tf.equal(n_max, 2):
            i += -1
        path_input = path_input.write(compteur, i - 1)
        path_output = path_output.write(compteur, j - 1)
    return path_input.stack()[:-1][::-1], path_output.stack()[:-1][::-1]

# Classique convolution
def convolution_1D(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 1, 0])
    output_list = []
    for j in range(0, inputs.shape[-2] - weights.shape[-1] + 1):
        output_list.append(tf.linalg.trace(tf.linalg.matmul(inputs[j:j + weights.shape[-1]], weights)))
    output_final = tf.stack(output_list)
    return output_final

# permet de realigner les séries et réalise le calcul matriciel pour les poids chaque partie d'input
@tf.function
def slice_alignment(slice_input, weights, minmax='min'):
    output_list = tf.TensorArray(dtype=tf.float32, size=(weights.shape[0]))
    for filt in range(weights.shape[0]):
        # recuperation des "meilleurs chemins"
        t_weight = weights[filt]
        # Iwana DWA
        mat_allign = tf.sparse.to_dense(DTW_TF(slice_input, t_weight))
        # réalignement
        mat_allign = tf.reshape(mat_allign, (slice_input.shape[0], t_weight.shape[0]))
        output_list = output_list.write(filt, mat_allign)
    w_allign = output_list.stack() @ weights
    output = tf.math.reduce_sum(slice_input*weights, axis=[2,1])
    return output


# Fonction de base pour DWA de iwana
@tf.function
def conv1D_weight_alignment(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    output_final = tf.TensorArray(dtype=tf.float32, size=(inputs.shape[-2] - weights.shape[-2] + 1))
    for j in range(0, inputs.shape[-2] - weights.shape[-2] + 1):
        res = slice_alignment(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights)
        output_final = output_final.write(j, res)
    return output_final.stack()

# Fonction de base pour schulman cnn_dtw
def conv1D_schulman_dtw(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    tensor_iter = tf.constant([*range(0, inputs.shape[-2] - weights.shape[-2] + 1)])
    output_final = tf.map_fn(lambda j: slice_alignment(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights, 'max'),
                             tensor_iter, fn_output_signature=tf.float32)
    return output_final


## Buza & Antal
# Permet de récuperer les poids de la matrice dtw
def dtw_weight_return(slice_input, weights):
    output_list = []
    for filt in range(weights.shape[0]):
        t_input = slice_input
        t_weight = weights[filt]
        cost_dtw = DTW_TF(t_input, t_weight, minmaxpath='cost')
        output_list.append(cost_dtw)
    return tf.stack(output_list)


def conv1D_dynamic(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    tensor_iter = tf.constant([*range(0, inputs.shape[-2] - weights.shape[-2] + 1)])
    output_final = tf.map_fn(lambda j: dtw_weight_return(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights),
                             tensor_iter, fn_output_signature=tf.float32)
    return output_final
##


class CNN1D(tf.keras.layers.Conv1D):
    def call(self, inputs):
        output = tf.vectorized_map(lambda inp: convolution_1D(inp, self.kernel) + self.bias, inputs)
        return tf.nn.relu(output)

#Iwana
class DWA_CNN(tf.keras.layers.Conv1D):
    def call(self, inputs):
        output = tf.map_fn(lambda inp: conv1D_weight_alignment(inp, self.kernel) + self.bias, inputs)
        return tf.nn.relu(output)

## Buza & Antal
class DCNN(CNN1D):
    def call(self, inputs):
        output = tf.map_fn(lambda inp: conv1D_dynamic(inp, self.w) + self.b, inputs)
        return tf.nn.relu(output)

#Schulman
class DTW_CNN(CNN1D):
    def call(self, inputs):
        output = tf.map_fn(lambda inp: conv1D_schulman_dtw(inp, self.w) + self.b, inputs)
        return tf.nn.relu(output)

def dtw_path(s1, s2):
    if s1.shape[0] == 1:
        return np.ones([1,1]).astype("float32")
    dtw_f = dtw(s1,s2, step_pattern="symmetric1")
    mat_allign = np.zeros((s1.shape[0], s2.shape[0]))
    for ind in zip(dtw_f.index1, dtw_f.index2):
        mat_allign[ind] = 1
    return mat_allign.astype("float32")

@tf.function
def tf_function(t_input, t_weight):
    T = tf.numpy_function(dtw_path, (t_input, t_weight), [tf.dtypes.float32])
    return T

@tf.function
def slice_alignment_np(slice_input, weights):
    output_list = tf.TensorArray(dtype=tf.dtypes.float32, size=(weights.shape[0]))
    for filt in range(weights.shape[0]):
        t_weight = weights[filt]
        mat_allign = tf_function(slice_input, t_weight)
        mat_allign = tf.reshape(mat_allign, (slice_input.shape[0], t_weight.shape[0]))
        output_list = output_list.write(filt, mat_allign)
    w_allign = output_list.stack() @ weights
    output = tf.math.reduce_sum(slice_input*weights, axis=[2,1])
    return output

@tf.function
def conv1D_weight_alignment_np(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    output_final = tf.TensorArray(dtype=tf.dtypes.float32, size=(inputs.shape[-2] - weights.shape[-2] + 1))
    for j in range(0, inputs.shape[-2] - weights.shape[-2] + 1):
        res = slice_alignment_np(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights)
        output_final = output_final.write(j, res)
    return output_final.stack()

class DWA_CNN_np(tf.keras.layers.Conv1D):
    def call(self, inputs):
        output = tf.map_fn(lambda inp: conv1D_weight_alignment_np(inp, self.kernel) + self.bias, inputs)
        return tf.nn.relu(output)

if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, InputLayer, Flatten, Conv1D
    from tensorflow.keras.utils import to_categorical


    randi = np.random.random((100, 12, 3))
    y_train = np.random.randint(1, 3, 100)
    y_train = to_categorical(y_train)

    # Conv1D classic Test
    tf.random.set_seed(1234)
    model_conv = Sequential([
        InputLayer(randi.shape[1:]),
        Conv1D(5, 7, activation='relu', kernel_initializer='glorot_uniform'),
        Flatten(),
        Dense(3, activation="softmax")
    ])

    model_conv.summary()
    model_conv.compile(loss='categorical_crossentropy', metrics='accuracy')
    model_conv.fit(randi, y_train, epochs=20)

    # CNN1D Test
    tf.random.set_seed(1234)
    model_conv = Sequential([
        InputLayer(randi.shape[1:]),
        CNN1D(5, 7),
        Flatten(),
        Dense(3, activation="softmax")
    ])

    model_conv.summary()
    model_conv.compile(loss='categorical_crossentropy', metrics='accuracy')
    model_conv.fit(randi, y_train, epochs=20)

    # DWA Test
    tf.random.set_seed(1234)
    model = Sequential([
        InputLayer(randi.shape[1:]),
        DWA_CNN(5, 7),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', metrics='accuracy')
    model.fit(randi, y_train, epochs=5)

    #DWA_numpy
    tf.random.set_seed(1234)
    model = Sequential([
        InputLayer(randi.shape[1:]),
        DWA_CNN_np(5, 7),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', metrics='accuracy')
    model.fit(randi, y_train, epochs=10)

    # DCNN Test
    tf.random.set_seed(1234)
    model_conv = Sequential([
        InputLayer(randi.shape[1:]),
        DCNN(5, 3),
        Flatten(),
        Dense(3, activation="softmax")
    ])

    model_conv.summary()
    model_conv.compile(loss='categorical_crossentropy', metrics='accuracy')
    model_conv.fit(randi, y_train, epochs=20)

    # DTW_CNN Test
    tf.random.set_seed(1234)
    model_conv = Sequential([
        InputLayer(randi.shape[1:]),
        DTW_CNN(5, 4),
        Flatten(),
        Dense(3, activation="softmax")
    ])

    model_conv.summary()
    model_conv.compile(loss='categorical_crossentropy', metrics='accuracy')
    model_conv.fit(randi, y_train, epochs=20)

    S = [1, 3, 3, 3, 2, 0, 1]
    S = randi[1]
    S1 = randi[2]
    S1 = [0, 1, 3, 2, 2, 0, 1]

    # distance = dtw.distance(S, S1)
    # print(distance)
    # print(dtw.warping_paths(S, S1)[1])

    S = tf.convert_to_tensor(S)
    S1 = tf.convert_to_tensor(S1)

    DTW_TF(S, S1)
    dtw_path(S, S1)


    tf.random.set_seed(1234)
    model = Sequential([
        InputLayer(randi.shape[1:]),
        DWA_CNN_np(5, 7),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', metrics='accuracy')

    tf.random.set_seed(1234)
    model_tensor = Sequential([
        InputLayer(randi.shape[1:]),
        DWA_CNN(5, 7),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    model_tensor.summary()
    model_tensor.compile(loss='categorical_crossentropy', metrics='accuracy')


    weight_debut_dwa_numpy = model.weights[1]
    weight_debut_dwa_tensor = model_tensor.weights[1]

    model_tensor.fit(randi, y_train, epochs=1)
    model.fit(randi, y_train, epochs=1)