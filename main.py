"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import sys
from sklearn import preprocessing
import pandas as pd

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)
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

    #Robust scaler
    scaler = preprocessing.StandardScaler().fit(X_train)

    #robust scaling
    X_train_scale = scaler.transform(X_train)
    X_validation_scale = scaler.transform(X_validation)
    X_test_scale = scaler.transform(X_test)

    return X_train_scale, y_train, X_validation_scale, y_validation, X_test_scale


def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

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
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

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
    print_networks(networks[:3])

    # Write the network to file
    if i == generations - 1:
        networks[0].train_final_net(dataset_dict)
        networks[0].WriteModelToFile()
        networks[0].WriteResToFile(dataset_dict,"203768460_204380992_7.txt")


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
    generations = 4  # Number of times to evole the population.
    population = 3  # Number of networks in each generation.
    X_train, y_train, X_validation, y_validation, X_test = \
        load_process_data(train_file_name,valid_file_name,test_file_name)

    dataset_dict = { "X_train":X_train,
                        "y_train":y_train,
                        "X_validation":X_validation,
                        "y_validation":y_validation,
                        "X_test":X_test,
                        }


    nn_param_choices = {
        'Network_train_sample_size': [10000],
        'input_shape':[120],
        'batch_size':[32,64,128,256],
        'max_iter' :[200],
        'final_max_iter': [300],
        'hidden_layer_sizes': [64,128,256,384,512,1024,4096],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset_dict)

if __name__ == '__main__':
    train_file_name = sys.argv[1]
    validate_file_name = sys.argv[2]
    test_file_name = sys.argv[3]
    main(train_file_name,validate_file_name,test_file_name)


