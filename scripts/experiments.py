# Essential
import time
import sys, getopt
# DataScience tools
import numpy as np
import pandas as pd
# DL libraries
import mxnet as mx
import tensorflow as tf
# others
from generator import *



def main(argv):
    mode = 'cpu'
    verbose = False
    # dimension multiplier
    m_easy = 1  # 100 #mupltiplier for easy task (vector initialization)
    m_hard = 1  # 10 # multiplier for hard task (matrix multiplication)
    try:
        opts, args = getopt.getopt(argv,"hm:v:e:d:",["mode=","verbose=","multiplier_easy=", "multiplier_hard="])
    except getopt.GetoptError:
        print('test.py -m <cpu or gpu> -verbose <true or false> -e <1 if easy, 100 if hard> -d <1 if easy, 100 if hard>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -m <cpu or gpu> -verbose <true or false> -e <1 if easy, 100 if hard> -d <1 if easy, 10 if hard>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-e", "--multiplier_easy"):
            m_easy = int(arg)
        elif opt in ("-d", "--multiplier_hard"):
            m_hard = int(arg)
        elif opt in ("-v", "--verbose"):
            verbose = (arg == "true")
    print('Mode: ', mode)
    print('Verbose: ', verbose)
    print('m1 (multiplier for easy tasks): ', m_easy)
    print('m2 (multiplier for hard tasks): ', m_hard)

    print("Using the current versions. TF: ", tf.__version__, " - MX :", mx.__version__)
    print("Tensorflow - GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # GPU vs CPU
    if mode == "gpu":
        # set the connect for MXNet
        mx_context = mx.gpu()
        mx.test_utils.list_gpus()
    else:
        mx_context = mx.cpu()

    set_context(mx_context)

    # verbose
    if verbose:
        tf.debugging.set_log_device_placement(True)

    # EXPERIMENT 1 - VECTOR CREATION
    common_params = {}
    common_params["generate_vectors"] = "y"
    common_params["vectors_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}
    df_vector_creation = run_experiment_dimensions(mx_vector_creation, tf_vector_creation,
                                                                        start_dim=200*m_easy, bigger_dim=1000*m_easy,
                                                                        step_size=50*m_easy,
                                                                        nr_tries=20, tries_per_seed=5,
                                                                        common=common_params, mx_only=mx_params,
                                                                        tf_only=tf_params,
                                                                        verbose=False)
    df_vector_creation.to_csv("EX1-Vector-creation.csv")
    # EXPERIMENT 1 bis - VECTOR CREATION - FROM LIST
    common_params = {}
    common_params["generate_vectors"] = "y"
    common_params["vectors_per_bag"] = "50"
    common_params["numpy_vectors"] = "n"
    mx_params = {}
    mx_params["numpy_vectors"] = "n"
    tf_params = {}
    df_vector_creation_pylist = run_experiment_dimensions(mx_vector_creation, tf_vector_creation,
                                                                              start_dim=50*m_easy, bigger_dim=600*m_easy,
                                                                              step_size=50*m_easy,
                                                                              nr_tries=20, tries_per_seed=5,
                                                                              common=common_params, mx_only=mx_params,
                                                                              tf_only=tf_params,
                                                                              verbose=False)
    df_vector_creation_pylist.to_csv("EX1-bis-Vector-creation-python-list.csv", index=False)
    # EXPERIMENT 2 - Activation Functions
    common_params = {}
    common_params["generate_vectors"] = "y"
    common_params["vectors_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    common_params["need_of_premade_vectors"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}

    # EXPERIMENT 2 - A - SIGMOID
    common_params["activation"] = "sigmoid"
    df_act_sigmoid = run_experiment_dimensions(mx_apply_activation, tf_apply_activation,
                                                              start_dim=5*m_easy, bigger_dim=600*m_easy, step_size=30*m_easy,
                                                              nr_tries=20, tries_per_seed=5,
                                                              common=common_params, mx_only=mx_params,
                                                              tf_only=tf_params,
                                                              verbose=False)
    df_act_sigmoid.to_csv("EX2-A-Sigmoid.csv", index=False)
    # EXPERIMENT 2 - B - RELU
    common_params["activation"] = "relu"
    df_act_relu = run_experiment_dimensions(mx_apply_activation, tf_apply_activation,
                                                            start_dim=5 * m_easy, bigger_dim=600 * m_easy, step_size=30 * m_easy,
                                                            nr_tries=20, tries_per_seed=5,
                                                            common=common_params, mx_only=mx_params, tf_only=tf_params,
                                                            verbose=False)
    df_act_relu.to_csv("EX2-B-Relu.csv", index=False)
    # EXPERIMENT 2 - C - SOFTRELU
    common_params["activation"] = "softrelu"
    df_act_softrelu = run_experiment_dimensions(mx_apply_activation, tf_apply_activation,
                                                               start_dim=5* m_easy, bigger_dim=200* m_easy, step_size=10* m_easy,
                                                               nr_tries=20, tries_per_seed=5,
                                                               common=common_params, mx_only=mx_params,
                                                               tf_only=tf_params,
                                                               verbose=False)
    df_act_softrelu.to_csv("EX2-C-SoftRelu.csv", index=False)
    # EXPERIMENT 2 - D - TANH
    common_params["activation"] = "tanh"
    df_act_tanh = run_experiment_dimensions(mx_apply_activation, tf_apply_activation,
                                                           start_dim=5* m_easy, bigger_dim=600* m_easy, step_size=30* m_easy,
                                                           nr_tries=20, tries_per_seed=5,
                                                           common=common_params, mx_only=mx_params, tf_only=tf_params,
                                                           verbose=False)
    df_act_tanh.to_csv("EX2-D-Tanh.csv", index=False)
    # EXPERIMENT 2 - E - SOFT SIGN
    common_params["activation"] = "softsign"
    df_act_softsign = run_experiment_dimensions(mx_apply_activation, tf_apply_activation,
                                                               start_dim=5* m_easy, bigger_dim=600* m_easy, step_size=30* m_easy,
                                                               nr_tries=20, tries_per_seed=5,
                                                               common=common_params, mx_only=mx_params,
                                                               tf_only=tf_params,
                                                               verbose=False)
    df_act_softsign.to_csv("EX2-E-SoftSign.csv", index=False)
    # EXPERIMENT 3 - DOT PRODUCT
    common_params = {}
    common_params["generate_vectors"] = "y"
    common_params["vectors_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    common_params["need_of_premade_vectors"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}
    common_params["dot_product_per_trial"] = "50"
    df_dot_product = run_experiment_dimensions(mx_dot_product, tf_dot_product,
                                                              start_dim=10* m_easy, bigger_dim=1000* m_easy, step_size=100* m_easy,
                                                              nr_tries=20, tries_per_seed=5,
                                                              common=common_params, mx_only=mx_params,
                                                              tf_only=tf_params,
                                                              verbose=False)
    df_dot_product.to_csv("EX3-Dot-Product.csv", index=False)
    # EXPERIMENT 4 - Matrix Multiplication
    common_params = {}
    common_params["matrices_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    common_params["need_of_premade_matrices"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}
    common_params["mat_mul_per_trial"] = "10"
    df_mat_mul = run_experiment_dimensions(mx_matrix_multiplication, tf_matrix_multiplication,
                                                          start_dim=5*m_hard, bigger_dim=60*m_hard, step_size=5*m_hard,
                                                          nr_tries=20, tries_per_seed=5,
                                                          common=common_params, mx_only=mx_params, tf_only=tf_params,
                                                          verbose=False)
    df_mat_mul.to_csv("EX4-Matrix-Multiplication.csv", index=False)
    # EXPERIMENT 5 - Flatten Vector
    common_params = {}
    common_params["matrices_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    common_params["need_of_premade_matrices"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}
    df_flatten = run_experiment_dimensions(mx_flatten_square_matrix, tf_flatten_square_matrix,
                                                          start_dim=5*m_hard, bigger_dim=60*m_hard, step_size=5*m_hard,
                                                          nr_tries=20, tries_per_seed=5,
                                                          common=common_params, mx_only=mx_params, tf_only=tf_params,
                                                          verbose=False)
    df_flatten.to_csv("EX5-Flatten-Vector.csv", index=False)
    # EXPERIMENT 6 - Normalize
    common_params = {}
    common_params["matrices_per_bag"] = "50"
    common_params["numpy_vectors"] = "y"
    common_params["need_of_premade_matrices"] = "y"
    mx_params = {}
    mx_params["numpy_vectors"] = "y"
    tf_params = {}

    df_normalize = run_experiment_dimensions(mx_normalize_square_matrix, tf_normalize_square_matrix,
                                                            start_dim=5*m_hard, bigger_dim=50*m_hard, step_size=10*m_hard,
                                                            nr_tries=20, tries_per_seed=3,
                                                            common=common_params, mx_only=mx_params, tf_only=tf_params,
                                                            verbose=False)
    df_normalize.to_csv("EX6-Normalize-Matrix.csv", index=False)


if __name__ == "__main__":
   main(sys.argv[1:])