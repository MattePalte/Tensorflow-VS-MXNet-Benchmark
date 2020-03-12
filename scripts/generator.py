import numpy as np
import mxnet as mx
import tensorflow as tf
import pandas as pd
from typing import List, Tuple, Callable, Mapping
import time

mx_context = mx.cpu()

def set_context(ctx):
    global mx_context
    mx_context = ctx

def generate_Vectors(seed : int, dimensions : int, nr_vectors : int, numpy : bool = True):
    # set the seed
    np.random.seed(seed)
    # generate a set of vectors in numpy of the same dimensions
    bag_of_vectors = []
    for _ in range(nr_vectors):
        if numpy == True:
            bag_of_vectors += [np.random.rand(1, dimensions)]
        else:
            bag_of_vectors += [list(np.random.rand(1, dimensions))]
    #print(bag_of_vectors)
    return bag_of_vectors

def generate_Matrices(seed : int, dim_vertical : int, dim_horizontal : int, nr_matrices : int, numpy : bool = True):
    # set the seed
    np.random.seed(seed)
    # generate a set of vectors in numpy of the same dimensions
    bag_of_matrices  = []
    for _ in range(nr_matrices):
        if numpy == True:
            bag_of_matrices += [np.random.rand(dim_vertical,dim_horizontal)]
        else:
            bag_of_matrices += [list(np.random.rand(dim_vertical, dim_horizontal))]
    return bag_of_matrices


def run_experiment(mx_fun: Callable, tf_fun: Callable,
                   nr_tries: int, tries_per_seed: int,
                   common: Mapping[str, str],
                   mx_only: Mapping[str, str] = None, tf_only: Mapping[str, str] = None,
                   verbose=False):
    # mx_only and tf_only are passed directly to the two functions

    mx_timing = []
    tf_timing = []
    seed_list = []

    for seed in range(nr_tries):
        # enrich the parameters dictionary (common, mx_only, tf_only)
        # with some additional info
        if "generate_vectors" in common.keys():
            if common["generate_vectors"] == "y":
                # generate vectors in numpy
                numpy = (common["numpy_vectors"] == "y")
                dimensions = int(common["vector_dim"])
                nr_vectors_per_bag = int(common["vectors_per_bag"])
                bag_of_vectors = generate_Vectors(seed, dimensions, nr_vectors_per_bag, numpy)
                common["vectors"] = bag_of_vectors

        # if the benchmark function require vectors for their computation
        # you should pass a common dictionary that contains "need_of_premade_vectors" = y
        # then very function will find those vectors inside its own dictionary at
        # the beginning of the computation with the "name pre_made_vectors"
        # NB the time of vector creation is not counted for the benchmarking
        if "need_of_premade_vectors" in common.keys():
            if common["need_of_premade_vectors"] == "y":
                # initialize ndarray vector for mxnet
                mx_vectors = mx_vector_creation(common, mx_only)
                mx_only["pre_made_vectors"] = mx_vectors
                # initialize tensors vector for tensorflow
                tf_vectors = tf_vector_creation(common, tf_only)
                tf_only["pre_made_vectors"] = tf_vectors

        # create matrices if needed
        if "need_of_premade_matrices" in common.keys():
            if common["need_of_premade_matrices"] == "y":
                dimensions = int(common["vector_dim"])
                nr_matrices = int(common["matrices_per_bag"])
                bag_of_matrices = generate_Matrices(seed, dimensions, dimensions, nr_matrices)
                common["vectors"] = bag_of_matrices
                # initialize ndarray vector for mxnet
                mx_matrices = mx_vector_creation(common, mx_only)
                mx_only["pre_made_matrices"] = mx_matrices
                # initialize tensors vector for tensorflow
                tf_matrices = tf_vector_creation(common, tf_only)
                tf_only["pre_made_matrices"] = tf_matrices

        # create a list of pairs indices that will identify the vectors that will
        # go in the dot product
        # select the pairs of vectors that we will multiply
        if "dot_product_per_trial" in common.keys():
            indices = np.random.randint(len(bag_of_vectors), size=int(common["dot_product_per_trial"]) * 2)
            pairs = np.split(indices, int(common["dot_product_per_trial"]))
            common["pair_of_indices"] = pairs

        # create a list of pairs indices that will identify the matrices that will
        # go in the matrix product
        # select the pairs of vectors that we will multiply
        if "mat_mul_per_trial" in common.keys():
            indices = np.random.randint(len(bag_of_matrices), size=int(common["mat_mul_per_trial"]) * 2)
            pairs = np.split(indices, int(common["mat_mul_per_trial"]))
            common["pair_of_indices"] = pairs

        print(seed, end="-(") if verbose else 0

        for sub_try in range(tries_per_seed):
            # save the seed to record which pack of vector was used
            # for this subtry
            seed_list += [seed]
            print(".", end="") if verbose else 0
            # mxnet
            start_mx = time.process_time()
            mx_fun(common, mx_only)
            end_mx = time.process_time()
            elapsed_mx = end_mx - start_mx
            mx_timing += [elapsed_mx]

            # tensorflow
            start_tf = time.process_time()
            tf_fun(common, tf_only)
            end_tf = time.process_time()
            elapsed_tf = end_tf - start_tf
            tf_timing += [elapsed_tf]

        print(")-", end="") if verbose else 0

    mx_s = pd.Series(mx_timing)
    tf_s = pd.Series(tf_timing)
    seed_s = pd.Series(seed_list)
    dim_s = pd.Series([dimensions] * len(seed_list))

    df = pd.concat([mx_s, tf_s, seed_s, dim_s], axis=1)
    df = df.rename(columns={0: "mxnet", 1: "tensorflow", 2: "seed", 3: "dim"})

    return df


def run_experiment_dimensions(mx_fun: Callable, tf_fun: Callable,
                              start_dim: int, bigger_dim: int, step_size: int,
                              nr_tries: int, tries_per_seed: int, common: Mapping[str, str],
                              mx_only: Mapping[str, str] = None, tf_only: Mapping[str, str] = None,
                              verbose=False) -> pd.DataFrame:
    # you have to pass the following params
    '''
    common_params = {}
    common_params["generate_vectors"] = "y"
    common_params["vectors_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}
    '''

    pandas_dataframes = []

    # changing dimension of vectors
    for dimension in range(start_dim, bigger_dim, step_size):
        common["vector_dim"] = str(dimension)
        # print("Start dim:", dimension)
        df_dimension_result = run_experiment(mx_fun, tf_fun,
                                             nr_tries=nr_tries, tries_per_seed=tries_per_seed,
                                             common=common, mx_only=mx_only, tf_only=tf_only, verbose=verbose)
        pandas_dataframes += [df_dimension_result]
    # put together the results
    complete_df = pd.DataFrame()
    for df in pandas_dataframes:
        complete_df = complete_df.append(df, ignore_index=True)

    return complete_df

# EXPERIMENT 1 - VECTOR CREATION

def mx_vector_creation(params_dict, params_mx = None):
    global mx_context
    # retrieve parameters
    bag_of_vectors = params_dict["vectors"]
    # start experiment
    mx_bag_of_vectors  = []
    for np_vector in bag_of_vectors:
        mx_bag_of_vectors += [mx.nd.array(np_vector, dtype='float32', ctx = mx_context)]
    return mx_bag_of_vectors

def tf_vector_creation(params_dict, params_tf = None):
    bag_of_vectors = params_dict["vectors"]
    tf_bag_of_vectors = []
    for np_vector in bag_of_vectors:
        tf_bag_of_vectors += [tf.convert_to_tensor(np_vector, dtype=tf.float32)]
    return tf_bag_of_vectors

# EXPERIMENT 2 - ACTIVATION FUNCTION
# define the two benchmarking functions
def mx_apply_activation(params_dict, params_mx = None):
    bag_of_vectors = params_mx["pre_made_vectors"]
    act_name = params_dict["activation"]
    mx_vectors_after_activation = []
    for input_vector in bag_of_vectors:
        mx_vectors_after_activation += [mx.ndarray.Activation(input_vector, act_type = act_name)]
    return mx_vectors_after_activation


def tf_apply_activation(params_dict, params_tf = None):
    bag_of_vectors = params_tf["pre_made_vectors"]
    act_name = params_dict["activation"]
    tf_vectors_after_activation = []
    for input_vector in bag_of_vectors:
        if act_name == "sigmoid":
            tf_vectors_after_activation += [tf.keras.activations.sigmoid(input_vector)]
        if act_name == "relu":
            tf_vectors_after_activation += [tf.keras.activations.relu(input_vector)]
        if act_name == "softsign":
            tf_vectors_after_activation += [tf.keras.activations.softsign(input_vector)]
        if act_name == "tanh":
            tf_vectors_after_activation += [tf.keras.activations.tanh(input_vector)]
        if act_name == "softrelu":
            tf_vectors_after_activation += [tf.keras.activations.softplus(input_vector)]
    return tf_vectors_after_activation

# EXPERIMENT 3 - DOT PRODUCT
# define the two benchmarking functions
def mx_dot_product(params_dict, params_mx = None):
    bag_of_vectors = params_mx["pre_made_vectors"]
    list_of_index_pairs  = params_dict["pair_of_indices"]
    # https://mxnet.incubator.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.dot
    # list_of_index_pairs contains pairs of integers
    # the first integer indicates the index (position) of the 1st vector in the bag_of_vectors
    # the first integer indicates the index (position) of the 2nd vector in the bag_of_vectors
    mx_dot_product_results  = []
    for (i_first, j_second) in list_of_index_pairs:
        first = bag_of_vectors[i_first][0]
        second = bag_of_vectors[j_second][0]
        mx_dot_product_results += [mx.ndarray.dot(first, second)]
    return mx_dot_product_results
def tf_dot_product(params_dict, params_tf = None):
    bag_of_vectors = params_tf["pre_made_vectors"]
    list_of_index_pairs = params_dict["pair_of_indices"]
    #https://www.tensorflow.org/api_docs/python/tf/tensordot
    tf_dot_product_results = []
    for (i_first, j_second) in list_of_index_pairs:
        first = bag_of_vectors[i_first][0]
        second = bag_of_vectors[j_second][0]
        tf_dot_product_results += [tf.tensordot(first, tf.transpose(second), axes=1)]
    return tf_dot_product_results

# EXPERIMENT 4 - MATRIX MULTIPLICATION
# define the two benchmarking functions
def mx_matrix_multiplication(params_dict, params_mx = None):
    #https://mxnet.incubator.apache.org/api/python/docs/api/ndarray/linalg/index.html#mxnet.ndarray.linalg.gemm2
    bag_of_matrices = params_mx["pre_made_matrices"]
    list_of_index_pairs = params_dict["pair_of_indices"]
    # list_of_index_pairs contains pairs of integers
    # the first integer indicates the index (position) of the 1st vector in the bag_of_vectors
    # the first integer indicates the index (position) of the 2nd vector in the bag_of_vectors
    mx_mat_mul_results  = []
    for (i_first, j_second) in list_of_index_pairs:
        first = bag_of_matrices[i_first]
        second = bag_of_matrices[j_second]
        mx_mat_mul_results += [mx.nd.linalg.gemm2(first, second)]
    #print("mx_mat_mul_results:", mx_mat_mul_results[0])
    return mx_mat_mul_results
def tf_matrix_multiplication(params_dict, params_tf = None):
    bag_of_matrices  = params_tf["pre_made_matrices"]
    list_of_index_pairs  = params_dict["pair_of_indices"]
    #https://www.tensorflow.org/api_docs/python/tf/tensordot
    tf_mat_mul_results  = []
    for (i_first, j_second) in list_of_index_pairs:
        first = bag_of_matrices[i_first]
        second = bag_of_matrices[j_second]
        tf_mat_mul_results += [tf.tensordot(first, second, axes = [[1], [0]])]
    #print("tf_mat_mul_results:", tf_mat_mul_results[0])
    return tf_mat_mul_results

# EXPERIMENT 5 - FLATTEN VECTOR
def mx_flatten_square_matrix(params_dict, params_mx = None):
    #https://mxnet.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.Reshape
    bag_of_matrices  = params_mx["pre_made_matrices"]
    mx_flattened_vectors= []
    for matrix in bag_of_matrices:
        mx_flattened_vectors += [matrix.reshape(shape=(-1,))]
    #print("Mxnet:", mx_flattened_vectors[0].shape)
    return mx_flattened_vectors
def tf_flatten_square_matrix(params_dict, params_tf = None):
    #https://www.tensorflow.org/api_docs/python/tf/reshape
    bag_of_matrices = params_tf["pre_made_matrices"]
    tf_flattened_vectors  = []
    for matrix in bag_of_matrices:
        tf_flattened_vectors += [tf.reshape(matrix, [-1])]
    #print("TF:",tf_flattened_vectors[0].shape)
    return tf_flattened_vectors

# EXPERIMENT 6 - NORMALIZE
def mx_normalize_square_matrix(params_dict, params_mx = None):
    #https://mxnet.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.L2Normalization
    bag_of_matrices = params_mx["pre_made_matrices"]
    mx_norm_matrices  = []
    for matrix in bag_of_matrices:
        new_m = matrix/mx.nd.norm(matrix, ord=2, axis=None)
        #print("mx_new_m: ", new_m.shape)
        mx_norm_matrices += [new_m]
    #print("Mxnet:", mx_norm_matrices[0])
    return mx_norm_matrices
def tf_normalize_square_matrix(params_dict, params_tf = None):
    #https://www.tensorflow.org/api_docs/python/tf/linalg/normalize
    bag_of_matrices  = params_tf["pre_made_matrices"]
    tf_norm_matrices  = []
    for matrix in bag_of_matrices:
        new_m = tf.linalg.normalize(matrix, ord=2, axis=None)[0]
        #print("tf_new_m: ", new_m.shape)
        tf_norm_matrices+= [new_m]
    #print("TF:",tf_norm_matrices[0])
    return tf_norm_matrices