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
from devol_help import DevolMain

TIME_STR = str(datetime.datetime.now()).replace(" ", "#")
FILE_NAME = TIME_STR+".txt"
MODEL_NAME = TIME_STR+".h5"

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

#### PARAM SECTION ###############3
###################################################################
generations = 1  # 14  # Number of times to evole the population.
population = 1  # 8 Number of networks in each generation.

nn_param_choices = {
    # 'Network_train_sample_size': [10000],
    'Network_train_sample_size': [1000],
    # 'batch_size':[16,32, 64, 128, 256, 512, 1024],
    # 'batch_size': [64,128,256, 512],
    'batch_size': [64],
    # 'hidden_layer_sizes': [64, 128, 256, 384, 512, 1024, 2048, 4096],
    'hidden_layer_sizes': [16],
    'max_iter': [10],
    'final_max_iter': [500],
}
###################################################################
###################################################################
###################################################################
def rows_scale(X_train, X_validation, X_test):

    X_train_4_f = np.stack(np.split(X_train, X_train.shape[1]/4, 1), 1)
    X_validation_4_f = np.stack(np.split(X_validation, X_validation.shape[1]/4, 1), 1)
    X_test_4_f = np.stack(np.split(X_test, X_test.shape[1]/4, 1), 1)

    for row in X_train_4_f:
        preprocessing.scale(row,copy=False)
    for row in X_validation_4_f:
        preprocessing.scale(row,copy=False)
    for row in X_test_4_f:
        preprocessing.scale(row,copy=False)
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

def feature_extraction(X_train, X_validation, X_test,subModelFeatures):
    n_f = 2 if subModelFeatures else 4
    # TODO tegular mean
    # X_train_mean = np.mean(np.stack(np.split(X_train, X_train.shape[1]/4, 1), 1), axis=1)
    # X_validation_mean = np.mean(np.stack(np.split(X_validation, X_validation.shape[1]/4, 1), 1), axis=1)
    # X_test_mean = np.mean(np.stack(np.split(X_test, X_test.shape[1]/4, 1), 1), axis=1)

    fc = 1.1
    first_apear = int(6-(X_train.shape[1] /n_f %2))

    weight_coeff = np.array([fc] * first_apear + [fc**2] * 6 + [fc**3] * 6 + [fc**4] * 6 + [fc**5] * 6)

    # fc = 1.014
    # arr_len = int(30-(X_train.shape[1] /n_f %2))
    # weight_coeff = np.array([fc**i for i in range(arr_len)])

    weight_coeff = weight_coeff.astype(float) / weight_coeff.sum()

    X_train_avg = np.average(weights=weight_coeff ,a=np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_avg = np.average(weights=weight_coeff ,a=np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_avg = np.average(weights=weight_coeff ,a=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_std = np.std(np.stack(np.split(X_train, X_train.shape[1]/n_f, 1), 1), axis=1)
    X_validation_std = np.std(np.stack(np.split(X_validation, X_validation.shape[1]/n_f, 1), 1), axis=1)
    X_test_std = np.std(np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1), axis=1)

    X_train_max = np.max(np.stack(np.split(X_train, X_train.shape[1]/n_f, 1), 1), axis=1)
    X_validation_max = np.max(np.stack(np.split(X_validation, X_validation.shape[1]/n_f, 1), 1), axis=1)
    X_test_max = np.max(np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1), axis=1)

    X_train_min = np.min(np.stack(np.split(X_train, X_train.shape[1]/n_f, 1), 1), axis=1)
    X_validation_min = np.min(np.stack(np.split(X_validation, X_validation.shape[1]/n_f, 1), 1), axis=1)
    X_test_min = np.min(np.stack(np.split(X_test, X_test.shape[1]/n_f, 1), 1), axis=1)

    # X_train = np.concatenate((X_train, X_train_avg, X_train_std, X_train_max), axis=1)
    # X_validation = np.concatenate(
    #     (X_validation, X_validation_avg, X_validation_std, X_validation_max), axis=1)
    # X_test = np.concatenate((X_test, X_test_avg, X_test_std, X_test_max), axis=1)

    X_train = np.concatenate((X_train,X_train_avg, X_train_std, X_train_min, X_train_max), axis=1)
    X_validation = np.concatenate((X_validation,X_validation_avg, X_validation_std, X_validation_min, X_validation_max), axis=1)
    X_test = np.concatenate((X_test,X_test_avg, X_test_std, X_test_min, X_test_max), axis=1)


    # X_train = np.concatenate((X_train_avg, X_train_std, X_train_min, X_train_max), axis=1)
    # X_validation = np.concatenate((X_validation_avg, X_validation_std, X_validation_min, X_validation_max), axis=1)
    # X_test = np.concatenate(( X_test_avg, X_test_std, X_test_min, X_test_max), axis=1)


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
        scaler = preprocessing.StandardScaler()
    elif scaler_type == 'Robust':
        scaler = preprocessing.RobustScaler()

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
    scaler_type = 'Standard'
    feature_extract = True
    subModelFeatures = False
    RowScale = False
    raw_n_feature=2



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

def TrainNetworkMultiprocess(network,dataset,shared_array,index):
    network.train(dataset)
    shared_array[index] = network.accuracy

def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    accuracy_Arr = multiprocessing.Array(c_double, len(networks))


    processes = []
    activated_network = set()
    for index,network in enumerate(networks):
        accuracy_Arr[index] = network.accuracy
        # for single process
        #network.train(dataset)
        #pbar.update(1)
        curr_net_param = network.network_params.items()
        tuple_curr_net_param  = tuple(curr_net_param)
        if tuple_curr_net_param in activated_network:
            header_note = "#"*80
            print(header_note)
            print(header_note)
            print(f"#### SKIP try to run an network that has already run {curr_net_param}")
            print(header_note)
            print(header_note)
            pbar.update(1)
            continue

        activated_network.add(tuple_curr_net_param)
        p = multiprocessing.Process(target=TrainNetworkMultiprocess,
                                    args=(network,dataset,accuracy_Arr,index))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
        pbar.update(1)

    pbar.close()

    # Update the accuracy, from the shared memory array
    for c_index, double_accur in enumerate(accuracy_Arr):
        logging.info(f"***The net before Update accuracy {networks[c_index].accuracy}")
        networks[c_index].accuracy = float(double_accur)
        logging.info(f"***The net after Update accuracy {networks[c_index].accuracy}")



def get_max_accuracy(networks):
    return max(x.accuracy for x in networks)

def get_average_accuracy(networks):
    total_accuracy = 0
    counted_net =0
    for network in networks:
        if network.accuracy != 0:
            counted_net+=1
        total_accuracy += network.accuracy
    #print(f"get_average_accuracy sum: {total_accuracy}")
    return total_accuracy / counted_net

def generate(generations, population, nn_param_choices, dataset_dict):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))
        print("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset_dict)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        max_accuracy = get_max_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info("Generation max: %.2f%%" % (max_accuracy * 100))

        logging.info('-'*80)
        print("***generation %d of %d*** score" %
              (i + 1, generations))
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        print("Generation max: %.2f%%" % (max_accuracy * 100))
        print('-' * 80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 3 networks.
    #print_networks(networks[:3])

    # Write the network to file
    if i == generations - 1:
        networks[0].train_final_net(dataset_dict)
        networks[0].WriteModelToFile()
        networks[0].WriteResToFile(dataset_dict,FILE_NAME)


def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main(train_file_name,valid_file_name,test_file_name,MyMain=True):
    """Evolve a network."""
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    X_train, y_train, X_validation, y_validation, X_test = \
        load_process_data(train_file_name, valid_file_name, test_file_name)

    dataset_dict = {"X_train": X_train,
                    "y_train": y_train,
                    "X_validation": X_validation,
                    "y_validation": y_validation,
                    "X_test": X_test,
                    }
    print("finish preprocessing")
    if MyMain:
        generate(generations, population, nn_param_choices, dataset_dict)
    else:
        DevolMain(dataset_dict,generations, population, MODEL_NAME,FILE_NAME)

if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name,False)


