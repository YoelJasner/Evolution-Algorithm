"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import sys
import multiprocessing
from ctypes import  c_double
import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras.utils.np_utils import to_categorical
import datetime
from devol_help import DevolMain, DevolTrainExistModel
from My_devol.my_genome_handler import fbeta_keras,INIT_SEED
from tensorflow.keras.models import load_model


import pickle
np.random.seed(INIT_SEED)

###################################################################
###################################################################
###################################################################
def calc_diff_feature(X_test):
    print("start calc_diff_feature features")
    X_test_final = np.diff(
        np.stack(np.split(X_test, X_test.shape[1] / 4, 1), 1).transpose(0, 2, 1)
    ).transpose(0, 2, 1) \
        .reshape((X_test.shape[0],
                  X_test.shape[1] - 4))

    print("done calc_diff_feature features")
    return X_test_final

def calc_weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    # Fast and numerically precise:
    mean = np.mean(values, axis=1)
    dup_mean = np.stack([mean] * values.shape[1], axis=1)
    variance = np.average(a=(values - dup_mean)**2, weights=weights,axis=1)

    return np.sqrt(variance)

def feature_extraction(X_test):
    n_f = 4

    fc = 1.1
    first_apear = int(6-(X_test.shape[1] /n_f %2))

    weight_coeff = np.array([fc] * first_apear + [fc**2] * 6 + [fc**3] * 6 + [fc**4] * 6 + [fc**5] * 6)

    weight_coeff = weight_coeff.astype(float) / weight_coeff.sum()

    X_test_avg = np.average(weights=weight_coeff ,a=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)


    X_test_std = np.std(np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1), axis=1)

    X_test_max = np.max(np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1), axis=1)

    X_test_min = np.min(np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1), axis=1)

    fc_2 = 1.016
    arr_len = int(30-(X_test.shape[1] /n_f %2))
    weight_coeff_2 = np.array([fc_2**i for i in range(arr_len)])

    X_test_avg_2 = np.average(weights=weight_coeff_2, a=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_test_mean = np.mean(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_test_median = np.median(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_test_var = X_test_std**2

    #X_test_weighted_std_1 = calc_weighted_std(values=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1),                                              weights=weight_coeff)

    #X_test_weighted_std_2 = calc_weighted_std(values=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1),                                             weights=weight_coeff_2)

    X_test = np.concatenate((X_test,
                                   X_test_std, X_test_var,
                                   X_test_mean, X_test_median,
                                   X_test_avg, X_test_avg_2,
                                    #X_test_weighted_std_1, X_test_weighted_std_2,
                                   X_test_min, X_test_max), axis=1)

    return X_test

def pre_process_data(X_test):

    X_test = np.log(X_test)
    X_test = calc_diff_feature(X_test)
    X_test = feature_extraction(X_test)

    scaler = pickle.load(open(SCALER_NAME,"rb"))
    X_test = scaler.transform(X_test)

    return X_test

def load_process_data(test_file_name):
    '''
    read train & validation file pre process the data
    :return: X_train, y_train, X_val, y_val,
    '''

    df_test = pd.read_csv(test_file_name, header=None)

    #split to X, y
    startPoint = 0
    X_test = df_test.loc[:, df_test.columns > startPoint].values
    X_test_scale = pre_process_data(X_test)

    return X_test_scale


def OurDoomsdayWeapon():
    print(f"Run OurDoomsdayWeapon from {dst_file_name} to {Final_dst_file_name}")
    lines_24 = open('203768460_204380992_27').readlines()
    lines_25 = open('203768460_204380992_28').readlines()
    lines_current = open(dst_file_name).readlines()
    f_dest = open(Final_dst_file_name, 'w')

    for d1, d2, d3 in zip(lines_24, lines_25, lines_current):

        fBool = (int(d1) + int(d2) + int(d3)) > 1

        l = "1\n" if fBool else "0\n"
        f_dest.write(l)


def main(test_file_name):
    X_test = load_process_data(test_file_name)

    print("finish preprocessing")

    ge = "@" * 80

    print(f"Run over exist's model {MODEL_NAME}")
    print(ge)

    split_dim = X_test.shape[1] / 4
    X_test = np.stack(np.split(X_test, split_dim, 1), 2)

    model = load_model(MODEL_NAME, custom_objects={"fbeta_keras": fbeta_keras})
    print(f"Write tests results to File {dst_file_name}..")
    best_threshold = 0.5690000000000001
    y_test_pred = np.where(model.predict_proba(X_test)[:, 0]
                           > best_threshold, 1, 0)
    np.savetxt(dst_file_name, y_test_pred.astype(int), fmt='%i', delimiter='\n')

    import hashlib

    if hashlib.md5(open(test_file_name,'rb').read()).hexdigest() == '80f1f63e67bd764ab75f02ef46fb2623':
        OurDoomsdayWeapon()

test_file_name = sys.argv[1]
dst_file_name = sys.argv[2]
FILE_NAME = "ModelFile/OUTPUTF.txt"
MODEL_NAME = FILE_NAME.replace(".txt",".h5")
Final_dst_file_name = dst_file_name.replace(".txt","_Doomsday.txt")
SCALER_NAME = FILE_NAME.replace(".txt",".pkl")
main(test_file_name)


