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
def DTW_TF(S, S1, d=euclideanDistance, only_cost=False):
    cost_matrix = tf.TensorArray(dtype=tf.float32, size=S.shape[0]+1)
    cost_matrix = cost_matrix.write(0, tf.cast(tf.stack([0, *([np.inf] * S1.shape[0])]), "float32"))
    for i in tf.range(1,S.shape[0]+1):
        sub_cost_j = tf.TensorArray(dtype=tf.float32, size=S1.shape[0]+1)
        sub_cost_j = sub_cost_j.write(0, np.inf)
        for j in tf.range(1, S1.shape[0]+1):
            dst = d(S[i-1], S1[j-1])
            mat_dt =[
                dst + sub_cost_j.stack()[j-1],
                dst + cost_matrix.stack()[i-1, j-1],
                dst + cost_matrix.stack()[i-1, j]
            ]
            sub_cost_j = sub_cost_j.write(j, tf.reduce_min(mat_dt))
        cost_matrix = cost_matrix.write(i, sub_cost_j.stack())
    if only_cost:
        return cost_matrix.stack()[-1, -1]
    return cost_matrix.stack()



# Classique convolution
class CNN1D(tf.keras.layers.Conv1D):
    def call(self, inputs):
        i = 0
        output = tf.TensorArray(dtype=tf.dtypes.float32, size=(tf.shape(inputs)[0]))
        inputs = tf.expand_dims(inputs, 1)
        for inp in tf.image.extract_patches(images=inputs,
                                            sizes=[1, 1, self.kernel_size[0], 1],
                                            strides=[1, 1, *self.strides, 1],
                                            rates=[1, 1, 1, 1],
                                            padding=self.padding.upper()):
            w_a = tf.transpose(self.kernel, perm=[2, 1, 0])
            inp = tf.reshape(inp, tf.TensorShape((inp.shape[1], *w_a.shape[1:][::-1])))
            output = output.write(i, tf.vectorized_map(lambda inp_i : tf.linalg.trace(inp_i@w_a) + self.bias, inp))
            i+=1
        return tf.nn.relu(output.stack())

#Iwana
class DWA_CNN(tf.keras.layers.Conv1D):
    # Renvoie le chemin dtw optimal en 2 fonctions
    @tf.function
    def loop_function(self, i, j, best_path, compteur, cost_mat):
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
    def DTW_minimal_path(self, S, S1):
        cost_mat = DTW_TF(S, S1)
        i = cost_mat.shape[0] - 1
        j = cost_mat.shape[1] - 1
        compteur = 0
        best_path = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        best_path = best_path.write(compteur, (i-1, j-1))
        cond = lambda i, j, best_path, compteur, cost_mat: tf.logical_and(tf.greater(i, 0), tf.greater(j, 0))
        i, j, best_path, compteur, cost_mat = tf.while_loop(cond, self.loop_function, [i, j, best_path, compteur, cost_mat])
        best_path = tf.cast(best_path.stack()[:-1][::-1], tf.int64)

        mat_allign = tf.sparse.SparseTensor(indices=best_path,
                                            values=tf.ones(tf.shape(best_path)[0], dtype=tf.dtypes.float32),
                                            dense_shape=[cost_mat.shape[0] - 1, cost_mat.shape[1] - 1])
        return mat_allign

    # permet de realigner les séries et réalise le calcul matriciel pour les poids chaque partie d'input
    @tf.function
    def slice_alignment(self, slice_input, weights, minmax='min'):
        output_list = tf.TensorArray(dtype=tf.float32, size=(weights.shape[0]))
        for filt in range(weights.shape[0]):
            # recuperation des "meilleurs chemins"
            t_weight = weights[filt]
            # Iwana DWA
            mat_allign = tf.sparse.to_dense(self.DTW_minimal_path(slice_input, t_weight))
            # réalignement
            mat_allign = tf.reshape(mat_allign, (slice_input.shape[0], t_weight.shape[0]))
            output_list = output_list.write(filt, mat_allign)
        w_allign = output_list.stack() @ weights
        output = tf.linalg.trace(tf.matmul(slice_input,w_allign, transpose_b=True))
        return output

    def call(self, inputs):
        i = 0
        output = tf.TensorArray(dtype=tf.dtypes.float32, size=(tf.shape(inputs)[0]))
        inputs = tf.expand_dims(inputs, 1)
        for inp in tf.image.extract_patches(images=inputs,
                                            sizes=[1, 1, self.kernel_size[0], 1],
                                            strides=[1, 1, *self.strides, 1],
                                            rates=[1, 1, 1, 1],
                                            padding=self.padding.upper()):
            w_a = tf.transpose(self.kernel, perm=[2, 0, 1])
            inp = tf.reshape(inp, tf.TensorShape((inp.shape[1], *w_a.shape[1:])))
            output = output.write(i, tf.map_fn(lambda inp_i : self.slice_alignment(inp_i, w_a) + self.bias, inp))
            i += 1
        return tf.nn.relu(output.stack())

## Buza & Antal
class DCNN(CNN1D):
    ## Buza & Antal
    # Permet de récuperer les poids de la matrice dtw
    def dtw_weight_return(self, slice_input, weights):
        output_list = []
        for filt in range(weights.shape[0]):
            t_weight = weights[filt]
            cost_dtw = DTW_TF(slice_input, t_weight, only_cost=True)
            output_list.append(cost_dtw)
        return tf.stack(output_list)

    def call(self, inputs):
        i = 0
        output = tf.TensorArray(dtype=tf.dtypes.float32, size=(tf.shape(inputs)[0]))
        inputs = tf.expand_dims(inputs, 1)
        for inp in tf.image.extract_patches(images=inputs,
                                            sizes=[1, 1, self.kernel_size[0], 1],
                                            strides=[1, 1, *self.strides, 1],
                                            rates=[1, 1, 1, 1],
                                            padding=self.padding.upper()):
            w_a = tf.transpose(self.kernel, perm=[2, 0, 1])
            inp = tf.reshape(inp, tf.TensorShape((inp.shape[1], *w_a.shape[1:])))
            output = output.write(i, tf.map_fn(lambda inp_i : self.dtw_weight_return(inp_i, w_a) + self.bias, inp))
            i += 1
        return tf.nn.relu(output.stack())

#Schulman
class DTW_CNN(CNN1D):
    # renvoie le chemin qui maximise la matrice DTW pour Schulman
    @tf.function
    def loop_function(self, i, j, best_path, compteur, cost_mat):
        compteur += 1
        cost_min = tf.stack([
            cost_mat[i-1, j-1],
            cost_mat[i, j-1],
            cost_mat[i-1, j]
        ])
        n_max = tf.math.argmax(cost_min)
        i, j = tf.case([
            (tf.equal(n_max,0), lambda: (i-1, j-1)),
            (tf.equal(n_max,1), lambda: (i, j-1)),
            (tf.equal(n_max,2), lambda: (i-1, j))
        ])
        best_path = best_path.write(compteur, (i-1, j-1))
        return i, j, best_path, compteur, cost_mat

    @tf.function
    def DTW_maximal_path(self, S, S1):
        cost_mat = DTW_TF(S, S1, scalarDistance)
        i = cost_mat.shape[0] - 1
        j = cost_mat.shape[1] - 1
        compteur = 0
        best_path = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        best_path = best_path.write(compteur, (i-1, j-1))
        cond = lambda i, j, best_path, compteur, cost_mat: tf.logical_and(tf.greater(i, 0), tf.greater(j, 0))
        i, j, best_path, compteur, cost_mat = tf.while_loop(cond, self.loop_function, [i, j, best_path, compteur, cost_mat])
        best_path = tf.cast(best_path.stack()[:-1][::-1], tf.int64)
        mat_allign = tf.sparse.SparseTensor(indices=best_path,
                                            values=tf.ones(tf.shape(best_path)[0], dtype=tf.dtypes.float32),
                                            dense_shape=[cost_mat.shape[0] - 1, cost_mat.shape[1] - 1])
        return mat_allign

    # permet de realigner les séries et réalise le calcul matriciel pour les poids chaque partie d'input
    @tf.function
    def slice_alignment(self, slice_input, weights):
        output_list = tf.TensorArray(dtype=tf.float32, size=(weights.shape[0]))
        for filt in range(weights.shape[0]):
            # recuperation des "meilleurs chemins"
            t_weight = weights[filt]
            # Iwana DWA
            mat_allign = tf.sparse.to_dense(self.DTW_maximal_path(slice_input, t_weight))
            # réalignement
            mat_allign = tf.reshape(mat_allign, (slice_input.shape[0], t_weight.shape[0]))
            output_list = output_list.write(filt, mat_allign)
        w_allign = output_list.stack() @ weights
        output = tf.linalg.trace(tf.matmul(slice_input,w_allign, transpose_b=True))
        return output

    def call(self, inputs):
        i = 0
        output = tf.TensorArray(dtype=tf.dtypes.float32, size=(tf.shape(inputs)[0]))
        inputs = tf.expand_dims(inputs, 1)
        for inp in tf.image.extract_patches(images=inputs,
                                            sizes=[1, 1, self.kernel_size[0], 1],
                                            strides=[1, 1, *self.strides, 1],
                                            rates=[1, 1, 1, 1],
                                            padding=self.padding.upper()):
            w_a = tf.transpose(self.kernel, perm=[2, 0, 1])
            inp = tf.reshape(inp, tf.TensorShape((inp.shape[1], *w_a.shape[1:])))
            output = output.write(i, tf.map_fn(lambda inp_i : self.slice_alignment(inp_i, w_a) + self.bias, inp))
            i += 1
        return tf.nn.relu(output.stack())


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
    output = tf.math.reduce_sum(slice_input*w_allign, axis=[2,1])
    return output

@tf.function
def conv1D_weight_alignment_np(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    output_final = tf.TensorArray(dtype=tf.dtypes.float32, size=(inputs.shape[-2] - weights.shape[-2] + 1))
    for j in tf.range(0, inputs.shape[-2] - weights.shape[-2] + 1):
        res = slice_alignment_np(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights)
        output_final = output_final.write(j, res)
    return output_final.stack()

class DWA_CNN_np(tf.keras.layers.Conv1D):
    def call(self, inputs):
        i = 0
        output = tf.TensorArray(dtype=tf.dtypes.float32, size=(tf.shape(inputs)[0]))
        inputs = tf.expand_dims(inputs, 1)
        for inp in tf.image.extract_patches(images=inputs,
                                            sizes=[1, 1, self.kernel_size[0], 1],
                                            strides=[1, 1, *self.strides, 1],
                                            rates=[1, 1, 1, 1],
                                            padding=self.padding.upper()):
            w_a = tf.transpose(self.kernel, perm=[2, 0, 1])
            inp = tf.reshape(inp, tf.TensorShape((inp.shape[1], *w_a.shape[1:])))
            output = output.write(i, tf.map_fn(lambda inp_i : slice_alignment_np(inp_i, w_a) + self.bias, inp))
            i += 1
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
        Conv1D(5, 7, 3, activation='relu', kernel_initializer='glorot_uniform'),
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
        CNN1D(5, 7, 3, kernel_initializer='glorot_uniform'),
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
        DTW_CNN(5, 4, 1),
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