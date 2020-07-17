from __future__ import print_function
from keras.utils.np_utils import to_categorical
import numpy as np
from devol import DEvol, GenomeHandler
from sklearn import preprocessing
import pandas as pd

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

def feature_extraction(X_train, X_validation, X_test):
    X_train_mean = np.mean(np.stack(np.split(X_train, 30, 1), 1), axis=1)
    X_validation_mean = np.mean(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
    X_test_mean = np.mean(np.stack(np.split(X_test, 30, 1), 1), axis=1)

    X_train_std = np.std(np.stack(np.split(X_train, 30, 1), 1), axis=1)
    X_validation_std = np.std(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
    X_test_std = np.std(np.stack(np.split(X_test, 30, 1), 1), axis=1)

    X_train_max = np.max(np.stack(np.split(X_train, 30, 1), 1), axis=1)
    X_validation_max = np.max(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
    X_test_max = np.max(np.stack(np.split(X_test, 30, 1), 1), axis=1)

    X_train_min = np.min(np.stack(np.split(X_train, 30, 1), 1), axis=1)
    X_validation_min = np.min(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
    X_test_min = np.min(np.stack(np.split(X_test, 30, 1), 1), axis=1)

    X_train = np.concatenate((X_train_mean, X_train_std, X_train_min, X_train_max), axis=1)
    X_validation = np.concatenate((X_validation_mean, X_validation_std, X_validation_min, X_validation_max), axis=1)
    X_test = np.concatenate((X_test_mean, X_test_std, X_test_min, X_test_max), axis=1)

    return X_train, X_validation, X_test


def pre_process_data(X_train, X_validation, X_test,
                     scaler_type, feature_extract=True, log_scale=True):

    if log_scale:
        X_train = np.log10(X_train)
        X_validation = np.log10(X_validation)
        X_test = np.log10(X_test)

    #create scaler
    if scaler_type == 'Standard':
        scaler = preprocessing.StandardScaler().fit(X_train)
    elif scaler_type == 'Robust':
        scaler = preprocessing.RobustScaler().fit(X_train)

    # robust scaling
    X_train_scale = scaler.transform(X_train)
    X_validation_scale = scaler.transform(X_validation)
    X_test_scale = scaler.transform(X_test)

    if feature_extract == True:
        X_train_scale, X_validation_scale, X_test_scale = feature_extraction(X_train_scale, X_validation_scale, X_test_scale)

    return X_train_scale[:100000, :], X_validation_scale[:100000, :], X_test_scale[:100000, :]

def load_process_data(train_file_name,valid_file_name,test_file_name):
    '''
    read train & validation file pre process the data
    :return: X_train, y_train, X_val, y_val,
    '''
    df_train = pd.read_csv(train_file_name, header=None)
    df_validation = pd.read_csv(valid_file_name, header=None)
    df_test = pd.read_csv(test_file_name, header=None)

    #split to X, y
    X_train = df_train.loc[:, df_train.columns != 0]
    y_train = df_train.loc[:, df_train.columns == 0].values
    X_validation = df_validation.loc[:, df_validation.columns != 0]
    y_validation = df_validation.loc[:, df_validation.columns == 0].values
    X_test = df_test.loc[:, df_validation.columns != 0]

    X_train_scale, X_validation_scale, X_test_scale = \
        pre_process_data(X_train, X_validation, X_test, 'Standard')

    return X_train_scale, y_train[:100000, :], X_validation_scale, y_validation[:100000, :], X_test_scale

X_train, y_train, X_validation, y_validation, _ = load_process_data('train.csv','validate.csv','test.csv')
y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)
dataset = ((X_train, y_train), (X_validation, y_validation))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

genome_handler = GenomeHandler(max_conv_layers=0,
                               max_dense_layers=9, # includes final dense layer
                               max_filters=256,
                               max_dense_nodes=2048,
                               input_shape=X_train.shape[1:],
                               n_classes=2)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.
# The best model is returned decoded and with `epochs` training done.

devol = DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=8,
                  pop_size=10,
                  epochs=200)
print(model.summary())
