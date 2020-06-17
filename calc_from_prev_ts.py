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



def calc_diff_feature(X_train, X_validation, X_test):
    print("start diff features")
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

    print("done diff features")
    return X_train_final, X_validation_final, X_test_final


def main(train_file_name,valid_file_name,test_file_name):
    '''
    read train & validation file pre process the data
    :return: X_train, y_train, X_val, y_val,
    '''
    df_train = pd.read_csv(train_file_name, header=None)
    df_validation = pd.read_csv(valid_file_name, header=None)
    df_test = pd.read_csv(test_file_name, header=None)

    #split to X, y
    X_train = df_train.loc[:, df_train.columns != 0].values
    y_train = df_train.loc[:, df_train.columns == 0].values
    X_validation = df_validation.loc[:, df_validation.columns != 0].values
    y_validation = df_validation.loc[:, df_validation.columns == 0].values
    X_test = df_test.loc[:, df_validation.columns != 0].values
    y_test = np.asarray([[-1] * X_test.shape[0]]).T
    X_train_prev_ts, X_validation_prev_ts, X_test_prev_ts = \
        calc_diff_feature(X_train, X_validation, X_test)

    train_table = np.concatenate((y_train, X_train_prev_ts), axis=1)
    validation_table = np.concatenate((y_validation, X_validation_prev_ts), axis=1)
    test_table = np.concatenate((y_test, X_test_prev_ts), axis=1)

    np.savetxt(train_file_name.replace(".csv","_diff.csv"), train_table, delimiter=",",fmt='%1.6f')
    np.savetxt(valid_file_name.replace(".csv", "_diff.csv"), validation_table, delimiter=",",fmt='%1.6f')
    np.savetxt(test_file_name.replace(".csv", "_diff.csv"), test_table, delimiter=",",fmt='%1.6f')



if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name)


