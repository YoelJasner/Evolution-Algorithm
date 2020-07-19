"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import sys
from sklearn import preprocessing
import pandas as pd
import multiprocessing
from ctypes import  c_double

import numpy as np

def rows_scale(X_train, X_validation, X_test,log_scale=False,n_f=4):
    print("start rows_scale features")

    if log_scale:
        X_train = np.log(X_train)
        X_validation = np.log(X_validation)
        X_test = np.log(X_test)

    X_train_4_f = np.stack(np.split(X_train, X_train.shape[1]/n_f, 1), 1)
    X_validation_4_f = np.stack(np.split(X_validation, X_validation.shape[1]/n_f, 1), 1)
    X_test_4_f = np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1)

    for row in X_train_4_f:
        preprocessing.scale(row,copy=False,axis=1)
    for row in X_validation_4_f:
        preprocessing.scale(row,copy=False,axis=1)
    for row in X_test_4_f:
        preprocessing.scale(row,copy=False,axis=1)
    X_train_4_f = X_train_4_f.reshape((X_train.shape))
    X_validation_4_f = X_validation_4_f.reshape((X_validation.shape))
    X_test_4_f = X_test_4_f.reshape((X_test.shape))

    print("done rows_scale features")


    return X_train_4_f, X_validation_4_f, X_test_4_f


def feature_model_sub(X_train, X_validation, X_test):

    X_train = np.stack(np.split(X_train, X_train.shape[1]/4, 1), 1)
    X_validation = np.stack(np.split(X_validation, X_validation.shape[1]/4, 1), 1)
    X_test = np.stack(np.split(X_test, X_test.shape[1]/4, 1), 1)

    X_train = np.stack([X_train[:, :, 2] - X_train[:, :, 0], X_train[:, :, 3] - X_train[:, :, 1]], axis=1)
    X_train = X_train.transpose(0, 2, 1).reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

    X_validation = np.stack([X_validation[:, :, 2] - X_validation[:, :, 0], X_validation[:, :, 3] - X_validation[:, :, 1]], axis=1)
    X_validation = X_validation.transpose(0, 2, 1).reshape(X_validation.shape[0],  X_validation.shape[1] * X_validation.shape[2])

    X_test = np.stack([X_test[:, :, 2] - X_test[:, :, 0], X_test[:, :, 3] - X_test[:, :, 1]], axis=1)
    X_test = X_test.transpose(0, 2, 1).reshape(X_test.shape[0],   X_test.shape[1] * X_test.shape[2])

    return X_train, X_validation, X_test

def calc_diff_feature(X_train, X_validation, X_test,log_scale=False):
    print("start calc_diff_feature features")

    if log_scale:
        X_train = np.log(X_train)
        X_validation = np.log(X_validation)
        X_test = np.log(X_test)

    X_train_final = np.diff(
        np.stack(np.split(X_train, 30, 1), 1).transpose(0, 2, 1)
    ).transpose(0, 2, 1) \
        .reshape((X_train.shape[0],
                  X_train.shape[1] - 4))

    X_validation_final = np.diff(
        np.stack(np.split(X_validation, 30, 1), 1).transpose(0, 2, 1)
                                ).transpose(0, 2, 1)\
                                  .reshape((X_validation.shape[0],
                                            X_validation.shape[1] - 4))
    X_test_final = np.diff(
        np.stack(np.split(X_test, 30, 1), 1).transpose(0, 2, 1)
    ).transpose(0, 2, 1) \
        .reshape((X_test.shape[0],
                  X_test.shape[1] - 4))

    print("done calc_diff_feature features")
    return X_train_final, X_validation_final, X_test_final


def main(train_file_name,valid_file_name,test_file_name):
    '''
    read train & validation file pre process the data
    :return: X_train, y_train, X_val, y_val,
    '''
    log_scale=False
    diff_feature=False
    row_scale=True
    subModelFeatures=False
    RawScaleOverModel = True
    raw_num_of_feature=2 if RawScaleOverModel else 4

    ac_log_scale = log_scale

    df_train = pd.read_csv(train_file_name, header=None)
    df_validation = pd.read_csv(valid_file_name, header=None)
    df_test = pd.read_csv(test_file_name, header=None)

    #split to X, y
    X_train = df_train.loc[:, df_train.columns != 0].values
    y_train = df_train.loc[:, df_train.columns == 0].values
    X_validation = df_validation.loc[:, df_validation.columns != 0].values
    y_validation = df_validation.loc[:, df_validation.columns == 0].values
    X_test = df_test.loc[:, df_test.columns != 0].values
    y_test = np.asarray([[-1] * X_test.shape[0]]).T

    if row_scale and diff_feature:
        prefix=f"rows_{raw_num_of_feature}_diff"
    else:
        prefix = f"rows_{raw_num_of_feature}" if row_scale else "diff"


    if diff_feature:
        X_train_prev_ts, X_validation_prev_ts, X_test_prev_ts = \
             calc_diff_feature(X_train, X_validation, X_test,ac_log_scale)
        ac_log_scale = False
    if row_scale:
        X_train_prev_ts, X_validation_prev_ts, X_test_prev_ts = \
            rows_scale(X_train, X_validation, X_test, ac_log_scale,raw_num_of_feature)
        ac_log_scale=False

    train_table = np.concatenate((y_train, X_train_prev_ts), axis=1)
    validation_table = np.concatenate((y_validation, X_validation_prev_ts), axis=1)
    test_table = np.concatenate((y_test, X_test_prev_ts), axis=1)

    if log_scale:
        sufix = f"_log_{prefix}.csv"
    else:
        sufix = f"_{prefix}.csv"
    print(f"The new suffix:  ## {sufix}")
    np.savetxt(train_file_name.replace(".csv",sufix), train_table, delimiter=",",fmt='%1.6f')
    np.savetxt(valid_file_name.replace(".csv", sufix), validation_table, delimiter=",",fmt='%1.6f')
    np.savetxt(test_file_name.replace(".csv", sufix), test_table, delimiter=",",fmt='%1.6f')



if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name)


