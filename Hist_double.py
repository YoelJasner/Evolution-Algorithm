"""Entry point to evolving the neural network. Start here."""
import logging
import sys
import numpy as np
from sklearn import preprocessing
import pandas as pd
import datetime
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score,make_scorer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from gpMY import get_best_threshold
TIME_STR = str(datetime.datetime.now()).replace(" ", "#")
FILE_NAME = "hist_"+ TIME_STR+".txt"
MODEL_NAME = "hist_" + TIME_STR+".h5"

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


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
    if log_scale:
        X_train = np.log(X_train)
        X_validation = np.log(X_validation)
        X_test = np.log(X_test)

    # TODO: Note that i replace the feature_extraction input's from
    #  regular to _scale
    if feature_extract == True:
        X_train, X_validation, X_test = \
            feature_extraction(X_train,
                               X_validation,
                               X_test,
                               subModelFeatures)
    # TODO: remove if needed
    #return X_train, X_validation, X_test

    # create scaler
    if scaler_type == 'Standard':
        scaler = preprocessing.StandardScaler()
    elif scaler_type == 'Robust':
        scaler = preprocessing.RobustScaler()
    elif scaler_type == 'MinMax':
        scaler = preprocessing.MinMaxScaler()

    scaler.fit(X_train)
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

    bFeatureDiff = False
    log_scale = True
    RowScale = True

    subModelFeatures = False
    scaler_type = 'Standard'
    feature_extract = True

    RawScaleOverModel = True
    raw_n_feature=2 if RawScaleOverModel else 4



    if bFeatureDiff and RowScale:
        # print("cant do feature diff also raw scale")
        # exit(-1)
        print("Run both feature diff also raw scale")
        prefix = f"rows_{raw_n_feature}_diff"
    else:
        prefix = f"rows_{raw_n_feature}" if RowScale else "diff"

    if bFeatureDiff or RowScale:
        sufix = f"_{prefix}.csv"
        if log_scale:
            sufix = f"_log_{prefix}.csv"


        # In any case the log already calculate
        log_scale = False

        train_file_name= train_file_name.replace(".csv",sufix)
        valid_file_name = valid_file_name.replace(".csv", sufix)
        test_file_name = test_file_name.replace(".csv", sufix)


    df_train = pd.read_csv(train_file_name, header=None)
    df_validation = pd.read_csv(valid_file_name, header=None)
    df_test = pd.read_csv(test_file_name, header=None)

    #split to X, y
    startPoint = 0
    X_train = df_train.loc[:, df_train.columns > startPoint].values
    y_train = df_train.loc[:, df_train.columns == 0].values.astype(int)
    X_validation = df_validation.loc[:, df_validation.columns > startPoint].values
    y_validation = df_validation.loc[:, df_validation.columns == 0].values.astype(int)
    X_test = df_test.loc[:, df_test.columns > startPoint].values

    X_train_scale, X_validation_scale, X_test_scale = \
        pre_process_data(X_train, X_validation, X_test,
                         scaler_type=scaler_type,
                         feature_extract=feature_extract,
                         subModelFeatures=subModelFeatures,
                         RowScale=RowScale,
                         log_scale=log_scale)

    return X_train_scale, y_train, X_validation_scale, y_validation, X_test_scale

def main(train_file_name,valid_file_name,test_file_name):
    X_train, y_train, X_validation, y_validation, X_test = \
        load_process_data(train_file_name, valid_file_name, test_file_name)


    print("finish preprocessing")
    f_scorer_full = make_scorer(fbeta_score, beta=0.125)
    f_scorer_model1 = make_scorer(fbeta_score, beta=0.09)
    f_scorer_model2 = make_scorer(fbeta_score, beta=0.09)
    f_scorer_model3 = make_scorer(fbeta_score, beta=0.09)
    f_scorer_model4 = make_scorer(fbeta_score, beta=0.09)
    MAX_D = 6
    MAX_ITER = 100
    L_R_full = 0.06
    L_R = 0.07
    N_ITER = 8
    V_F = None

    full_model = HistGradientBoostingClassifier(scoring=f_scorer_full,
                                                #max_depth=4,
                                                max_iter=MAX_ITER,
                                                learning_rate=L_R_full,
                                                #n_iter_no_change=10,
                                                validation_fraction = None,
                                                verbose=2)

    model_1 = HistGradientBoostingClassifier(scoring=f_scorer_model1,
                                             # max_depth=MAX_D,
                                             max_iter=MAX_ITER,
                                             learning_rate=L_R,
                                             n_iter_no_change=N_ITER,
                                             validation_fraction=V_F,
                                             verbose=0)

    model_2 = HistGradientBoostingClassifier(scoring=f_scorer_model2,
                                                #max_depth=MAX_D,
                                                max_iter=MAX_ITER,
                                                learning_rate=L_R,
                                                n_iter_no_change=N_ITER,
                                                validation_fraction = V_F,
                                                verbose=0)

    model_3 = HistGradientBoostingClassifier(scoring=f_scorer_model3,
                                             # max_depth=MAX_D,
                                             max_iter=MAX_ITER,
                                             learning_rate=L_R,
                                             n_iter_no_change=N_ITER,
                                             validation_fraction=V_F,
                                             verbose=0)
    model_4 = HistGradientBoostingClassifier(scoring=f_scorer_model4,
                                             # max_depth=MAX_D,
                                             max_iter=MAX_ITER,
                                             learning_rate=L_R,
                                             n_iter_no_change=N_ITER,
                                             validation_fraction=V_F,
                                             verbose=0)
    # X_train_4f = np.stack(np.split(X_train, X_train.shape[1] / 4, 1), 1)
    # X_validation_4f = np.stack(np.split(X_validation, X_validation.shape[1] / 4, 1), 1)
    # X_test_4f = np.stack(np.split(X_test, X_test.shape[1] / 4, 1), 1)

    # Split the dataset
    # X_train_model1 = np.stack([X_train_4f[:, :, 0] , X_train_4f[:, :, 2]],axis=1).\
    #     transpose(0, 2, 1).\
    #     reshape(X_train.shape[0], int(X_train.shape[1]/2))
    # X_train_model2 = np.stack([X_train_4f[:, :, 1] , X_train_4f[:, :, 3]],axis=1).\
    #     transpose(0, 2, 1).\
    #     reshape(X_train.shape[0], int(X_train.shape[1]/2))
    # X_train_model3 = np.stack([X_train_4f[:, :, 0], X_train_4f[:, :, 1]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_train.shape[0], int(X_train.shape[1] / 2))
    # X_train_model4 = np.stack([X_train_4f[:, :, 2], X_train_4f[:, :, 3]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_train.shape[0], int(X_train.shape[1] / 2))
    #
    # X_validation_model1 = \
    #     np.stack([X_validation_4f[:, :, 0], X_validation_4f[:, :, 2]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_validation.shape[0], int(X_validation.shape[1] / 2))
    # X_validation_model2 = \
    #     np.stack([X_validation_4f[:, :, 1], X_validation_4f[:, :, 3]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_validation.shape[0], int(X_validation.shape[1] / 2))
    # X_validation_model3 = \
    #     np.stack([X_validation_4f[:, :, 0], X_validation_4f[:, :, 1]], axis=1). \
    #         transpose(0, 2, 1). \
    #         reshape(X_validation.shape[0], int(X_validation.shape[1] / 2))
    # X_validation_model4 = \
    #     np.stack([X_validation_4f[:, :, 2], X_validation_4f[:, :, 3]], axis=1). \
    #         transpose(0, 2, 1). \
    #         reshape(X_validation.shape[0], int(X_validation.shape[1] / 2))
    #
    # X_test_model1 = np.stack([X_test_4f[:, :, 0], X_test_4f[:, :, 2]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_test.shape[0], int(X_test.shape[1] / 2))
    # X_test_model2 = np.stack([X_test_4f[:, :, 1], X_test_4f[:, :, 3]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_test.shape[0], int(X_test.shape[1] / 2))
    # X_test_model3 = np.stack([X_test_4f[:, :, 0], X_test_4f[:, :, 1]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_test.shape[0], int(X_test.shape[1] / 2))
    # X_test_model4 = np.stack([X_test_4f[:, :, 2], X_test_4f[:, :, 3]], axis=1). \
    #     transpose(0, 2, 1). \
    #     reshape(X_test.shape[0], int(X_test.shape[1] / 2))

    dataset_full_model = {"X_train": X_train,
                    "y_train": y_train,
                    "X_validation": X_validation,
                    "y_validation": y_validation,
                    "X_test": X_test,
                    }
    # dataset_model1 = {"X_train": X_train_model1,
    #                 "y_train": y_train,
    #                 "X_validation": X_validation_model1,
    #                 "y_validation": y_validation,
    #                 "X_test": X_test_model1,
    #                 }
    # dataset_model2 = {"X_train": X_train_model2,
    #                   "y_train": y_train,
    #                   "X_validation": X_validation_model2,
    #                   "y_validation": y_validation,
    #                   "X_test": X_test_model2,
    #                   }
    # dataset_model3 = {"X_train": X_train_model3,
    #                   "y_train": y_train,
    #                   "X_validation": X_validation_model3,
    #                   "y_validation": y_validation,
    #                   "X_test": X_test_model3,
    #                   }
    # dataset_model4 = {"X_train": X_train_model4,
    #                   "y_train": y_train,
    #                   "X_validation": X_validation_model4,
    #                   "y_validation": y_validation,
    #                   "X_test": X_test_model4,
    #                   }
    model_list = []
    model_list.append((full_model,dataset_full_model,"Full"))
    # model_list.append((model_1, dataset_model1, "model1"))
    # model_list.append((model_2, dataset_model2, "model2"))
    # model_list.append((model_3, dataset_model3, "model3"))
    # model_list.append((model_4, dataset_model4, "model4"))

    y_val_list=[]
    y_test_list=[]

    str_header = "$" * 78

    for c_model, c_data_set,c_name in model_list:
        print(f"Running {c_name}")
        c_model.fit(c_data_set["X_train"], c_data_set["y_train"])

        y_val_proba = c_model.predict_proba(c_data_set["X_validation"])
        y_train_proba = c_model.predict_proba(c_data_set["X_train"])
        best_threshold = get_best_threshold(y_val_proba, c_data_set["y_validation"])

        y_train_pred = np.where(y_train_proba[:, 1]
                                > best_threshold, 1, 0)
        y_val_pred = np.where(y_val_proba[:, 1] > best_threshold, 1, 0)

        y_test_pred = np.where(c_model.predict_proba(c_data_set["X_test"])[:, 1]
                               > best_threshold, 1, 0)

        y_val_list.append(y_val_pred)
        y_test_list.append(y_test_pred)
        print(str_header)
        print(f"Using threshold {best_threshold}")
        print(str_header)
        print(c_name+ ' Train accuracy', accuracy_score(c_data_set["y_train"], y_train_pred))
        print(c_name+ ' Validation accuracy', accuracy_score(c_data_set["y_validation"], y_val_pred))

        print(c_name+ ' Train precision', precision_score(c_data_set["y_train"], y_train_pred))
        print(c_name+ ' Validation precision', precision_score(c_data_set["y_validation"], y_val_pred))

        print(c_name+ ' Train recall', recall_score(c_data_set["y_train"], y_train_pred))
        print(c_name+ ' Validation recall', recall_score(c_data_set["y_validation"], y_val_pred))

        print(c_name+ ' Train f-beta score', fbeta_score(c_data_set["y_train"], y_train_pred, beta=0.25))
        validation_beta_score = fbeta_score(c_data_set["y_validation"], y_val_pred, beta=0.25)
        print(f'{c_name} Validation f-beta score {validation_beta_score}')
        print(str_header)
        print(str_header)



    y_valid_pred_final = []
    for tup in zip(*y_val_list):
        s = np.sum(tup)
        l_en = len(tup) //2
        lab = 1 if s > l_en else 0
        y_valid_pred_final.append(lab)
    y_test_pred_final = []
    # The test step
    for tup in zip(*y_test_list):
        s = np.sum(tup)
        l_en = len(tup) // 2
        lab = 1 if s > l_en else 0
        y_test_pred_final.append(lab)

    str_header = "*" * 78
    c_name = "Merge_ALL"
    print(str_header)
    print(str_header)
    print(c_name + ' Validation accuracy', accuracy_score(y_validation, y_valid_pred_final))
    print(c_name + ' Validation precision', precision_score(y_validation, y_valid_pred_final))
    print(c_name + ' Validation recall', recall_score(y_validation, y_valid_pred_final))
    validation_beta_score = fbeta_score(y_validation, y_valid_pred_final, beta=0.25)
    print(f'{c_name} Validation f-beta score {validation_beta_score}')
    print(str_header)
    print(str_header)

    print(f"Write tests results to File {FILE_NAME}..")
    y_test_pred_arr = np.array(y_test_pred_final)
    np.savetxt(FILE_NAME, y_test_pred_arr.astype(int), fmt='%i', delimiter='\n')

if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name)


