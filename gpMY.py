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
from  gplearn.genetic import SymbolicClassifier
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
import numpy as np
from numpy import savetxt
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score

FILE_NAME = "GP_203768460_204380992_0.txt"

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def rows_scale(X_train, X_validation, X_test):

    X_train_4_f = np.stack(np.split(X_train, X_train.shape[1]/4, 1), 1)
    X_validation_4_f = np.stack(np.split(X_validation, X_validation.shape[1]/4, 1), 1)
    X_test_4_f = np.stack(np.split(X_test, X_test.shape[1]/4, 1), 1)

    for row in X_train_4_f:
        preprocessing.robust_scale(row,copy=False)
    for row in X_validation_4_f:
        preprocessing.robust_scale(row,copy=False)
    for row in X_test_4_f:
        preprocessing.robust_scale(row,copy=False)
    X_train_4_f = X_train_4_f.reshape((X_train.shape))
    X_validation_4_f = X_validation_4_f.reshape((X_validation.shape))
    X_test_4_f = X_test_4_f.reshape((X_test.shape))


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


def feature_extraction(X_train, X_validation, X_test, subModelFeatures):
    n_f = 2 if subModelFeatures else 4
    # TODO tegular mean
    # X_train_mean = np.mean(np.stack(np.split(X_train, X_train.shape[1]/4, 1), 1), axis=1)
    # X_validation_mean = np.mean(np.stack(np.split(X_validation, X_validation.shape[1]/4, 1), 1), axis=1)
    # X_test_mean = np.mean(np.stack(np.split(X_test, X_test.shape[1]/4, 1), 1), axis=1)

    fc = 1.1
    first_apear = int(6 - (X_train.shape[1] / n_f % 2))

    weight_coeff = np.array([fc] * first_apear + [fc ** 2] * 6 + [fc ** 3] * 6 + [fc ** 4] * 6 + [fc ** 5] * 6)

    weight_coeff = weight_coeff.astype(float) / weight_coeff.sum()

    X_train_avg = np.average(weights=weight_coeff, a=np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_avg = np.average(weights=weight_coeff,
                                  a=np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_avg = np.average(weights=weight_coeff, a=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_std = np.std(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_std = np.std(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_std = np.std(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_max = np.max(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_max = np.max(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_max = np.max(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_min = np.min(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_min = np.min(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_min = np.min(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    # X_train = np.concatenate((X_train,X_train_avg, X_train_std, X_train_min, X_train_max), axis=1)
    # X_validation = np.concatenate((X_validation,X_validation_avg, X_validation_std, X_validation_min, X_validation_max), axis=1)
    # X_test = np.concatenate((X_test,X_test_avg, X_test_std, X_test_min, X_test_max), axis=1)
    # return X_train, X_validation, X_test
    fc_2 = 1.016
    arr_len = int(30 - (X_train.shape[1] / n_f % 2))
    weight_coeff_2 = np.array([fc_2 ** i for i in range(arr_len)])

    X_train_avg_2 = np.average(weights=weight_coeff_2, a=np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1),
                               axis=1)
    X_validation_avg_2 = np.average(weights=weight_coeff_2,
                                    a=np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_avg_2 = np.average(weights=weight_coeff_2, a=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_mean = np.mean(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_mean = np.mean(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_mean = np.mean(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_median = np.median(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_median = np.median(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_median = np.median(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_quantile25 = np.quantile(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1, q=0.25)
    X_validation_quantile25 = np.quantile(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1,
                                          q=0.25)
    X_test_quantile25 = np.quantile(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1, q=0.25)

    X_train_quantile75 = np.quantile(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1, q=0.75)
    X_validation_quantile75 = np.quantile(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1,
                                          q=0.75)
    X_test_quantile75 = np.quantile(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1, q=0.75)

    X_train_var = X_train_std ** 2
    X_validation_var = X_validation_std ** 2
    X_test_var = X_test_std ** 2

    X_train_argmax = np.argmax(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_argmax = np.argmax(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_argmax = np.argmax(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_argmin = np.argmin(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_argmin = np.argmin(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_argmin = np.argmin(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train = np.concatenate((X_train,
                              X_train_std, X_train_var,
                              X_train_mean, X_train_median,
                              X_train_quantile25, X_train_quantile75,
                              X_train_avg, X_train_avg_2,
                              X_train_min, X_train_max), axis=1)
    X_validation = np.concatenate((X_validation,
                                   X_validation_std, X_validation_var,
                                   X_validation_mean, X_validation_median,
                                   X_validation_quantile25, X_validation_quantile75,
                                   X_validation_avg, X_validation_avg_2,
                                   X_validation_min, X_validation_max), axis=1)
    X_test = np.concatenate((X_test,
                             X_test_std, X_test_var,
                             X_test_mean, X_test_median,
                             X_test_quantile25, X_test_quantile75,
                             X_test_avg, X_test_avg_2,
                             X_test_min, X_test_max), axis=1)

    return X_train, X_validation, X_test

def pre_process_data(X_train, X_validation, X_test,
                     scaler_type, feature_extract,
                     log_scale, subModelFeatures,
                     RowScale):

    # Bad result using rowscale
    if False and RowScale:
        X_train, X_validation, X_test = \
            rows_scale(X_train,
                        X_validation,
                        X_test)

    # There is no case to log scale after minmaxRowScale
    elif log_scale:
        X_train = np.log(X_train)
        X_validation = np.log(X_validation)
        X_test = np.log(X_test)

    if subModelFeatures:
        X_train, X_validation, X_test = \
            feature_model_sub(X_train,
                              X_validation,
                              X_test)

    # TODO: Note that i replace the feature_extraction input's from
    #  regular to _scale
    if feature_extract == True:

        X_train, X_validation, X_test = \
            feature_extraction(X_train,
                               X_validation,
                               X_test,
                               subModelFeatures)

    # create scaler
    if scaler_type == 'Standard':
        scaler = preprocessing.StandardScaler().fit(X_train)
    elif scaler_type == 'Robust':
        scaler = preprocessing.RobustScaler().fit(X_train)

    # robust scaling
    X_train = scaler.transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    return X_train, X_validation, X_test

def load_process_data(train_file_name,valid_file_name,test_file_name):
    '''
    read train & validation file pre process the data
    :return: X_train, y_train, X_val, y_val,
    '''

    bFeatureDiff = True
    log_scale = True
    scaler_type = 'Standard'
    feature_extract = True
    subModelFeatures = False
    RowScale = False


    if bFeatureDiff:
        if log_scale:
            train_file_name= train_file_name.replace(".csv","_log_diff.csv")
            valid_file_name = valid_file_name.replace(".csv", "_log_diff.csv")
            test_file_name = test_file_name.replace(".csv", "_log_diff.csv")
            log_scale=False
        else:
            train_file_name = train_file_name.replace(".csv", "_diff.csv")
            valid_file_name = valid_file_name.replace(".csv", "_diff.csv")
            test_file_name = test_file_name.replace(".csv", "_diff.csv")

    df_train = pd.read_csv(train_file_name, header=None)
    df_validation = pd.read_csv(valid_file_name, header=None)
    df_test = pd.read_csv(test_file_name, header=None)

    #split to X, y
    startPoint = 0
    X_train = df_train.loc[:, df_train.columns > startPoint].values
    y_train = df_train.loc[:, df_train.columns == 0].values
    X_validation = df_validation.loc[:, df_validation.columns > startPoint].values
    y_validation = df_validation.loc[:, df_validation.columns == 0].values
    X_test = df_test.loc[:, df_test.columns > startPoint].values

    X_train_scale, X_validation_scale, X_test_scale = \
        pre_process_data(X_train, X_validation, X_test,
                         scaler_type=scaler_type,
                         feature_extract=feature_extract,
                         subModelFeatures=subModelFeatures,
                         RowScale=RowScale,
                         log_scale=log_scale)

    return X_train_scale, y_train, X_validation_scale, y_validation, X_test_scale

def get_best_threshold(y_val_proba, y_validation):
    best_threshold = 0.5
    best_fbeta_score_valid = 0
    beta = 0.25
    for threshold in np.arange(0.5, 0.8, 0.001):
        y_val_pred = np.where(y_val_proba[:, 1] > threshold, 1, 0)

        curr_validation_beta_score = fbeta_score(y_validation, y_val_pred, beta=beta)

        if curr_validation_beta_score >= best_fbeta_score_valid:# and curr_train_beta_score >= best_fbeta_score_train:

            best_fbeta_score_valid = curr_validation_beta_score
            best_threshold = threshold
    return best_threshold

def main(train_file_name,valid_file_name,test_file_name):

    X_train, y_train, X_validation, y_validation, X_test = \
        load_process_data(train_file_name, valid_file_name, test_file_name)

    gp_classifier = SymbolicClassifier(population_size=20,
                                       generations=65,
                                       tournament_size=3,
                                       const_range=None,
                                       init_depth=(4, 12),
                                       parsimony_coefficient=0.00000000000000000000000000000001,
                                       # parsimony_coefficient=0.0,
                                       # init_method='full',
                                       function_set=('add', 'sub',
                                                     'mul', 'div'),
                                       # make_function(my_sqr, "sqr", arity=2, wrap=False)),
                                       transformer='sigmoid',
                                       #metric=f_beta,
                                       p_crossover=0.85,
                                       p_subtree_mutation=0.04,
                                       p_hoist_mutation=0.01,
                                       p_point_mutation=0.04,
                                       p_point_replace=0.005,
                                       max_samples=1.0,
                                       feature_names=None,
                                       warm_start=False,
                                       low_memory=True,
                                       n_jobs=8,
                                       verbose=1,
                                       random_state=None)


    gp_classifier.fit(X_train, y_train)

    y_val_proba = gp_classifier.predict_proba(X_validation)
    y_train_proba = gp_classifier.predict_proba(X_train)
    best_threshold = get_best_threshold(y_val_proba, y_validation)

    y_train_pred = np.where(y_train_proba[:, 1]
                            > best_threshold, 1, 0)
    y_val_pred = np.where(y_val_proba[:, 1] > best_threshold, 1, 0)
    str_header = "$"*78
    print(str_header)
    print(str_header)
    print('Train accuracy', accuracy_score(y_train, y_train_pred))
    print('Validation accuracy', accuracy_score(y_validation, y_val_pred))

    print('Train precision', precision_score(y_train, y_train_pred))
    print('Validation precision', precision_score(y_validation, y_val_pred))

    print('Train recall', recall_score(y_train, y_train_pred))
    print('Validation recall', recall_score(y_validation, y_val_pred))

    print('Train f-beta score', fbeta_score(y_train, y_train_pred, beta=0.25))
    validation_beta_score = fbeta_score(y_validation, y_val_pred, beta=0.25)
    print(f'Validation f-beta score {validation_beta_score}')
    print(str_header)
    print(str_header)


    # y_val_pred = np.where(y_val_proba[:, 1] > best_threshold, 1, 0)
    #
    # final_test_predict_y = gp_classifier.predict(X_test)
    # np.savetxt(FILE_NAME, final_test_predict_y.astype(int), fmt='%i', delimiter='\n')


if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name)


