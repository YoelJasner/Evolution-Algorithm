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
import pickle


# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

#### PARAM SECTION ###############3
###################################################################
generations = 8  #   # Number of times to evole the population.
population = 10  #  Number of networks in each generation.

nn_param_choices = {
    'Network_train_sample_size': [1000],
    'batch_size': [64],
    'hidden_layer_sizes': [16],
    'max_iter': [10],
    'final_max_iter': [500],
}
###################################################################
###################################################################
###################################################################
def calc_diff_feature(X_train, X_validation, X_test):
    print("start calc_diff_feature features")

    X_train_final = np.diff(
        np.stack(np.split(X_train, X_train.shape[1] / 4, 1), 1).transpose(0, 2, 1)
    ).transpose(0, 2, 1) \
        .reshape((X_train.shape[0],
                  X_train.shape[1] - 4))

    X_validation_final = np.diff(
        np.stack(np.split(X_validation, X_validation.shape[1] / 4, 1), 1).transpose(0, 2, 1)
                                ).transpose(0, 2, 1)\
                                  .reshape((X_validation.shape[0],
                                            X_validation.shape[1] - 4))
    X_test_final = np.diff(
        np.stack(np.split(X_test, X_test.shape[1] / 4, 1), 1).transpose(0, 2, 1)
    ).transpose(0, 2, 1) \
        .reshape((X_test.shape[0],
                  X_test.shape[1] - 4))

    print("done calc_diff_feature features")
    return X_train_final, X_validation_final, X_test_final

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

def feature_extraction(X_train, X_validation, X_test,subModelFeatures):
    n_f = 2 if subModelFeatures else 4

    fc = 1.1
    first_apear = int(6-(X_train.shape[1] /n_f %2))

    weight_coeff = np.array([fc] * first_apear + [fc**2] * 6 + [fc**3] * 6 + [fc**4] * 6 + [fc**5] * 6)



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

    fc_2 = 1.016
    arr_len = int(30-(X_train.shape[1] /n_f %2))
    weight_coeff_2 = np.array([fc_2**i for i in range(arr_len)])

    X_train_avg_2 = np.average(weights=weight_coeff_2, a=np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_avg_2 = np.average(weights=weight_coeff_2,
                                  a=np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_avg_2 = np.average(weights=weight_coeff_2, a=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_mean = np.mean(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_mean = np.mean(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_mean = np.mean(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_median = np.median(np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1), axis=1)
    X_validation_median = np.median(np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1), axis=1)
    X_test_median = np.median(np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1), axis=1)

    X_train_var = X_train_std**2
    X_validation_var = X_validation_std**2
    X_test_var = X_test_std**2

    X_train_weighted_std_1 = calc_weighted_std(values=np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1),
                                              weights=weight_coeff)
    X_validation_weighted_std_1 = calc_weighted_std(values=np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1),
                                              weights=weight_coeff)
    X_test_weighted_std_1 = calc_weighted_std(values=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1),
                                              weights=weight_coeff)

    X_train_weighted_std_2 = calc_weighted_std(values=np.stack(np.split(X_train, X_train.shape[1] / n_f, 1), 1),
                                              weights=weight_coeff_2)
    X_validation_weighted_std_2 = calc_weighted_std(
        values=np.stack(np.split(X_validation, X_validation.shape[1] / n_f, 1), 1),
        weights=weight_coeff_2)
    X_test_weighted_std_2 = calc_weighted_std(values=np.stack(np.split(X_test, X_test.shape[1] / n_f, 1), 1),
                                             weights=weight_coeff_2)


    X_train = np.concatenate((X_train,
                              X_train_std,X_train_var,
                              X_train_mean,X_train_median,
                              X_train_avg, X_train_avg_2,
                             # X_train_weighted_std_1, X_train_weighted_std_2,
                              X_train_min, X_train_max), axis=1)
    X_validation = np.concatenate((X_validation,
                              X_validation_std,X_validation_var,
                              X_validation_mean, X_validation_median,
                              X_validation_avg, X_validation_avg_2,
                           #X_validation_weighted_std_1, X_validation_weighted_std_2,
                              X_validation_min, X_validation_max), axis=1)
    X_test = np.concatenate((X_test,
                                   X_test_std, X_test_var,
                                   X_test_mean, X_test_median,
                                   X_test_avg, X_test_avg_2,
                            #        X_test_weighted_std_1, X_test_weighted_std_2,
                                   X_test_min, X_test_max), axis=1)

    return X_train, X_validation, X_test

def pre_process_data(X_train, X_validation, X_test,
                     scaler_type, feature_extract,
                     log_scale, subModelFeatures,
                     RowScale,bFeatureDiff):

    # Bad result using rowscale
    if log_scale:
        X_train = np.log(X_train)
        X_validation = np.log(X_validation)
        X_test = np.log(X_test)

    if bFeatureDiff:
        X_train, X_validation, X_test = \
            calc_diff_feature(X_train,
                              X_validation,
                              X_test)

    if subModelFeatures:
        X_train, X_validation, X_test = \
            feature_model_sub(X_train,
                              X_validation,
                              X_test)

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
    pickle.dump(scaler, open(SCALER_NAME,"wb"))

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

    RawScaleOverModel = False
    raw_n_feature = 2 if RawScaleOverModel else 4

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
                         log_scale=log_scale,
                         bFeatureDiff=bFeatureDiff)

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

def main(train_file_name,valid_file_name,test_file_name,des):
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
    generate(generations, population, nn_param_choices, dataset_dict)

train_file_name = sys.argv[1]
validate_file_name = sys.argv[2]
test_file_name = sys.argv[3]
dst_file_name = sys.argv[4] # str(datetime.datetime.now()).replace(" ", "#")
FILE_NAME = dst_file_name
MODEL_NAME = dst_file_name.replace(".txt",".h5")
SCALER_NAME = dst_file_name.replace(".txt",".pkl")
main(train_file_name,validate_file_name,test_file_name,dst_file_name)

