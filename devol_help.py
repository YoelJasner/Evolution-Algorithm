from __future__ import print_function
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn import preprocessing
import pandas as pd
import logging
from optimizer import Optimizer
from tqdm import tqdm
import sys
import multiprocessing
from ctypes import  c_double
import numpy as np
from My_devol import MyDEvol, MyGenomeHandler
from sklearn import preprocessing
import pandas as pd
from keras.utils.np_utils import to_categorical
import datetime
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score


def get_best_threshold(y_val_proba, y_validation, y_train_proba, y_train):
    best_threshold = 0.5
    best_fbeta_score_valid = 0
    best_fbeta_score_train = 0
    beta = 0.25
    for threshold in np.arange(0.5, 0.8, 0.0025):
        y_val_pred = np.where(y_val_proba[:, 1] > threshold, 1, 0)
        y_train_pred = np.where(y_train_proba[:, 1] > threshold, 1, 0)

        curr_validation_beta_score = fbeta_score(y_validation, y_val_pred, beta=beta)
        curr_train_beta_score = fbeta_score(y_train, y_train_pred, beta=beta)

        if curr_validation_beta_score >= best_fbeta_score_valid:
            # if curr_train_beta_score >= best_fbeta_score_train:
            # and curr_train_beta_score >= best_fbeta_score_train:

            best_fbeta_score_valid = curr_validation_beta_score
            best_fbeta_score_train = curr_train_beta_score
            best_threshold = threshold

    header_note = "#" * 80
    print(header_note)
    print(f'#### improve thres:{best_threshold} With:')
    print(f'validation f-beta-{beta} score {best_fbeta_score_valid}')
    print(f'train f-beta-{beta} score {best_fbeta_score_train}')
    print(header_note)
    return best_threshold

def devol_train_final_model(model, dataset_dict):
    str_header = "#"*80
    print(str_header)
    print(f"best Network.. train_final_net  with param TODO:")
    print(str_header)

    dataset = ((dataset_dict['X_train'], dataset_dict['y_train']),
               (dataset_dict['X_validation'], dataset_dict['y_validation']))

    model.fit(dataset)

    y_val_proba = model.predict_proba(dataset_dict["X_validation"])
    y_train_proba = model.predict_proba(dataset_dict["X_train"])

    best_threshold = get_best_threshold(y_val_proba,
                               dataset_dict["y_validation"],
                               y_train_proba,
                               dataset_dict["y_train"])

    y_train_pred = np.where(y_train_proba[:, 1]
                            > best_threshold, 1, 0)
    y_val_pred = np.where(y_val_proba[:, 1] > best_threshold, 1, 0)
    print(str_header)
    print(str_header)
    print('Train accuracy', accuracy_score(dataset_dict["y_train"], y_train_pred))
    print('Validation accuracy', accuracy_score(dataset_dict["y_validation"], y_val_pred))

    print('Train precision', precision_score(dataset_dict["y_train"], y_train_pred))
    print('Validation precision', precision_score(dataset_dict["y_validation"], y_val_pred))

    print('Train recall', recall_score(dataset_dict["y_train"], y_train_pred))
    print('Validation recall', recall_score(dataset_dict["y_validation"], y_val_pred))

    print('Train f-beta score', fbeta_score(dataset_dict["y_train"], y_train_pred, beta=0.25))
    validation_beta_score = fbeta_score(dataset_dict["y_validation"], y_val_pred, beta=0.25)
    print(f'Validation f-beta score {validation_beta_score}')
    print(str_header)
    print(str_header)
    return model


def DevolMain(dataset_dict,generations,population,MODEL_NAME):
    # TODO: Delete after stableize
    generations=2
    population=2
    dataset_dict['X_train'] = dataset_dict['X_train'][:100, :]
    dataset_dict['y_train'] = dataset_dict['y_train'][:100, :]
    dataset_dict['X_validation'] = dataset_dict['X_validation'][:100, :]
    dataset_dict['y_validation'] = dataset_dict['y_validation'][:100, :]
    dataset_dict['X_test'] = dataset_dict['X_test'][:100, :]
    ## TODO: until this part..

    dataset_dict['y_train'] = to_categorical(dataset_dict['y_train'])
    dataset_dict['y_validation'] = to_categorical(dataset_dict['y_validation'])

    split_dim = 4#dataset_dict['X_train'].shape[1] / 4
    X_train_2_d = np.stack(np.split(dataset_dict['X_train'], split_dim , 1), 1)
    X_validation_2_d = np.stack(np.split(dataset_dict['X_validation'], split_dim, 1), 1)
    X_test_2_d = np.stack(np.split(dataset_dict['X_test'], split_dim, 1), 1)

    dataset = ((X_train_2_d[:10000, :], dataset_dict['y_train'][:10000, :]),
               (X_validation_2_d[:10000, :], dataset_dict['y_validation'][:10000, :]))

    print(X_train_2_d.shape)
    # **Prepare the genome configuration**
    # The `GenomeHandler` class handles the constraints that are imposed upon
    # models in a particular genetic program. See `genome-handler.py`
    # for more information.
    genome_handler = MyGenomeHandler(max_conv_layers=16,
                                   max_dense_layers=16,  # includes final dense layer
                                   max_filters=256,
                                   max_dense_nodes=2048,
                                   input_shape=X_train_2_d.shape[1:],
                                   n_classes=2)

    devol = MyDEvol(genome_handler)
    model = devol.run(dataset=dataset,
                      num_generations=generations,
                      pop_size=population,
                      epochs=200)

    model.save(MODEL_NAME)
    print(model.summary())