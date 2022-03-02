import tensorflow as tf

def euclideanDistance(x, y):
    dist = tf.sqrt(tf.cast(tf.reduce_sum(tf.square(x - y)), dtype="float32"))
    return dist

def scalarDistance(x,y):
    return x * y

#Fonction qui crée la matrice DTW
def DTW_TF(S, S1, d=euclideanDistance, minmaxpath="min"):
    cost_matrix = []
    cost_matrix.append(tf.cast(tf.stack([0, *([1234567891011] * S1.shape[0])]), "float32"))
    for i in range(1,S.shape[0]+1):
        sub_cost_j = [1234567891011]
        for j in range(1, S1.shape[0]+1):
            dst = d(S[i-1], S1[j-1])
            mat_dt = tf.stack([
            dst + sub_cost_j[j-1],
            dst + cost_matrix[i-1][j-1],
            dst + cost_matrix[i-1][j]
            ])
            sub_cost_j.append(tf.reduce_min(mat_dt))
        cost_matrix.append(tf.stack(sub_cost_j))
    if minmaxpath == 'min':
        return DTW_minimal_path(tf.stack(cost_matrix))
    elif minmaxpath == 'max':
        return DTW_maximal_path(tf.stack(cost_matrix))
    elif minmaxpath == 'cost':
        return cost_matrix[-1][-1]

# Renvoie le chemin dtw optimal
def DTW_minimal_path(cost_mat):
    i = cost_mat.shape[0] - 1
    j = cost_mat.shape[1] - 1
    compteur = 0
    path_input = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    path_input = path_input.write(compteur, i-1)
    path_output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    path_output = path_output.write(compteur, j-1)
    while tf.greater(i, 0) and tf.greater(j, 0):
        compteur += 1
        cost_min = tf.stack([
            cost_mat[i-1, j-1],
            cost_mat[i, j-1],
            cost_mat[i-1, j]
        ])
        n_min = tf.math.argmin(cost_min)
        if tf.equal(n_min,0):
            i += -1
            j += -1
        elif tf.equal(n_min,1):
            j += -1
        elif tf.equal(n_min, 2):
            i += -1
        path_input = path_input.write(compteur, i - 1)
        path_output = path_output.write(compteur, j - 1)
    return path_input.stack()[:-1][::-1], path_output.stack()[:-1][::-1]

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
def slice_alignment(slice_input, weights, minmax='min'):
    output_list = []
    for filt in range(weights.shape[0]):
        # recuperation des "meilleurs chemins"
        t_input = slice_input
        t_weight = weights[filt]
        # Iwana DWA
        if minmax == 'min':
            path_input, path_weight = DTW_TF(t_input, t_weight)
        # Schulman DTW_CNN
        elif minmax == 'max':
            path_input, path_weight = DTW_TF(t_input, t_weight, d=scalarDistance ,minmaxpath=minmax)
        else: raise ValueError
        # réalignement
        weights_align = tf.gather(t_weight, indices=path_weight)
        inputs_n = tf.gather(t_input, indices=path_input)
        weights_align = tf.linalg.matrix_transpose(weights_align)
        # Calcul des sorties
        output_list.append(tf.linalg.trace(tf.linalg.matmul(inputs_n, weights_align)))
    return tf.stack(output_list)


# Fonction de base pour DWA de iwana
def conv1D_weight_alignment(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    tensor_iter = tf.constant([*range(0, inputs.shape[-2] - weights.shape[-2] + 1)])
    output_final = tf.map_fn(lambda j: slice_alignment(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights),
                             tensor_iter, fn_output_signature=tf.float32)
    return output_final

# Fonction de base pour schulman cnn_dtw
def conv1D_schulman_dtw(inputs, weights):
    weights = tf.transpose(weights, perm=[2, 0, 1])
    tensor_iter = tf.constant([*range(0, inputs.shape[-2] - weights.shape[-2] + 1)])
    output_final = tf.map_fn(lambda j: slice_alignment(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights, 'max'),
                             tensor_iter, fn_output_signature=tf.float32)
    print(output_final)
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


class CNN1D(tf.keras.layers.Layer):
    def __init__(self, n_filters=8, kernel_size=3):
        super(CNN1D, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.b = self.add_weight(shape=(n_filters,), initializer="zeros", trainable=True)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.kernel_size, int(input_shape[-1]), self.n_filters),
            initializer="glorot_uniform", trainable=True
        )

    def call(self, inputs):
        output = tf.map_fn(lambda inp: convolution_1D(inp, self.w) + self.b, inputs)
        return tf.nn.relu(output)

#Iwana
class DWA_CNN(CNN1D):
    def call(self, inputs):
        output = tf.map_fn(lambda inp: conv1D_weight_alignment(inp, self.w) + self.b, inputs)
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
        Conv1D(5, 4, activation='relu', kernel_initializer='glorot_uniform'),
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
        CNN1D(5, 4),
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
        DWA_CNN(5, 4),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', metrics='accuracy')
    model.fit(randi, y_train, epochs=20)

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

    S = [1, 3, 3, 3, 2, 0, 0, 1]
    S = randi[1]
    S1 = randi[2]
    S1 = [0, 1, 3, 2, 2, 0, 1]

    # distance = dtw.distance(S, S1)
    # print(distance)
    # print(dtw.warping_paths(S, S1)[1])

    S = tf.convert_to_tensor(S)
    S1 = tf.convert_to_tensor(S1)

    DTW_TF(S, S1)


    #Test des numpy fonction, il semble avoir un problème de shape en retour
    def dtw_path(s1, s2):
        l1 = s1.shape[0]
        l2 = s2.shape[0]

        cum_sum = np.zeros((l1 + 1, l2 + 1))
        cum_sum[1:, 0] = np.inf
        cum_sum[0, 1:] = np.inf
        predecessors = [([None] * l2) for i in range(l1)]

        for i in range(l1):
            for j in range(l2):
                if np.isfinite(cum_sum[i + 1, j + 1]):
                    dij = np.linalg.norm(s1[i] - s2[j]) ** 2
                    pred_list = [cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]]
                    argmin_pred = np.argmin(pred_list)
                    cum_sum[i + 1, j + 1] = pred_list[argmin_pred] + dij
                    if i + j > 0:
                        if argmin_pred == 0:
                            predecessors[i][j] = (i - 1, j)
                        elif argmin_pred == 1:
                            predecessors[i][j] = (i, j - 1)
                        else:
                            predecessors[i][j] = (i - 1, j - 1)

        i = l1 - 1
        j = l2 - 1
        best_path = [(i, j)]
        while predecessors[i][j] is not None:
            i, j = predecessors[i][j]
            best_path.insert(0, (i, j))

        path_input, path_weight = zip(*best_path)
        return path_input, path_weight

    dtw_path(S, S1)

    def tf_function(t_input, t_weight):
      T = tf.numpy_function(dtw_path, (t_input, t_weight), [tf.int32, tf.int32])
      return T



    def slice_alignment(slice_input, weights, minmax='min'):
        output_list = []
        for filt in range(weights.shape[0]):
            t_input = slice_input
            t_weight = weights[filt]
            path_input, path_weight = tf_function(t_input, t_weight)
            weights_align = tf.gather(t_weight, indices=path_weight)
            inputs_n = tf.gather(t_input, indices=path_input)
            weights_align = tf.linalg.matrix_transpose(weights_align)
            output_list.append(tf.linalg.trace(tf.linalg.matmul(inputs_n, weights_align)))
        return tf.stack(output_list)


    @tf.function
    def conv1D_weight_alignment(inputs, weights):
        weights = tf.transpose(weights, perm=[2, 0, 1])
        tensor_iter = tf.constant([*range(0, inputs.shape[-2] - weights.shape[-2] + 1)])
        output_final = tf.map_fn(lambda j: slice_alignment(tf.slice(inputs, (j, 0), (weights.shape[-2:])), weights),
                                 tensor_iter, fn_output_signature=tf.float32)
        print(output_final)
        return output_final

    class DWA_CNN(CNN1D):
        @tf.function
        def call(self, inputs):
            output = tf.map_fn(lambda inp: conv1D_weight_alignment(inp, self.w) + self.b, inputs)
            return tf.nn.relu(output)

    tf.random.set_seed(1234)
    model = Sequential([
        InputLayer(randi.shape[1:]),
        DWA_CNN(5, 4),
        Flatten(),
        Dense(3, activation='softmax')
    ])