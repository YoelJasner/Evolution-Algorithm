"""Class that represents the network to be evolved."""
import random
import logging
from sklearn.neural_network import MLPClassifier
import numpy as np

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None,):
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network_params = {}  # (dic): represents MLP network parameters
        self.model = None

    def compile_model(self,bFinal=False):
        # Get our network parameters.
        max_iter = self.network_params['final_max_iter'] if bFinal else self.network_params['max_iter']
        self.model = MLPClassifier(max_iter=max_iter,
                                   verbose=2,
                                    batch_size=self.network_params["batch_size"],
                                   hidden_layer_sizes=self.network_params["hidden_layer_sizes"],)

    def create_random(self):
        for key in self.nn_param_choices:
            self.network_params[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        self.network_params = network

    def train(self, dataset_dict):
        if self.accuracy == 0.:
            self.accuracy = self.train_net(dataset_dict)

    def print_network(self):
        logging.info(self.network_params)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))


    def train_net(self, dataset_dict):
        self.compile_model(False)
        num_of_rows = self.network_params["Network_train_sample_size"]
        rows_index = np.random.choice(dataset_dict["X_train"].shape[0],
                                      size=num_of_rows,
                                      replace=False)

        self.model.fit(dataset_dict["X_train"][rows_index,:],
                  dataset_dict["y_train"][rows_index])

        score = self.model.score(dataset_dict["X_validation"], dataset_dict["y_validation"])

        return score

    def train_final_net(self, dataset_dict):
        print("train the best Network..")
        self.compile_model(bFinal=True)
        self.model.fit(dataset_dict["X_train"],
                       dataset_dict["y_train"])

        score = self.model.score(dataset_dict["X_validation"], dataset_dict["y_validation"])

        return score


    def WriteModelToFile(self):
        print("save net to model")
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
        print(self.network_params)
        self.print_network()
        # TODO: use pickle
        # self.model.save("model.h5")

    def WriteResToFile(self, ds_class,file_name):
        """Train the model, return test loss.

        Args:
            network (dict): the parameters of the network
            dataset (str): Dataset to use for training/evaluating

        """
        print(f"Write tests results to File {file_name}..")
        y_test_pred = self.model.predict(ds_class["X_test"])
        np.savetxt(file_name, y_test_pred.astype(int), fmt='%i', delimiter='\n')
