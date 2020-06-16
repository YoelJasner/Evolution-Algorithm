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

FILE_NAME = "203768460_204380992_10.txt"

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


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


def pre_process_data(X_train, X_validation, X_test, scaler_type, feature_extract=True, log_scale=True ):

    if log_scale:
        X_train = np.log(X_train)
        X_validation = np.log(X_validation)
        X_test = np.log(X_test)

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
        X_train_scale, X_validation_scale, X_test_scale = feature_extraction(X_train, X_validation, X_test)

    return X_train_scale, X_validation_scale, X_test_scale

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
        accuracy_Arr[index] = 0
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
    for c_index, network in enumerate(networks):
        network.accuracy = float(accuracy_Arr[c_index])

def get_max_accuracy(networks):
    return max(x.accuracy for x in networks)

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
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

def main(train_file_name,valid_file_name,test_file_name):
    """Evolve a network."""
    generations = 2 #14  # Number of times to evole the population.
    population = 3  #8 Number of networks in each generation.

    nn_param_choices = {
        'Network_train_sample_size': [10000],
        #'input_shape':[120],
        #'batch_size':[32, 64, 128, 256, 512, 1024],
        'batch_size': [16,32,64,128],
        #'hidden_layer_sizes': [64, 128, 256, 384, 512, 1024, 2048, 4096],
         'hidden_layer_sizes': [8,16,32,64,128],
        'max_iter' :[300],
        'final_max_iter': [500],

    }

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

    generate(generations, population, nn_param_choices, dataset_dict)

if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name)


