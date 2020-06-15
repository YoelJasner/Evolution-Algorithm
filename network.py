"""Class that represents the network to be evolved."""
import random
import logging
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score

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
        self.best_threshold = 0
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


    def update_best_threshold(self, y_val_proba, y_validation):
        self.best_threshold = 0
        best_fbeta_score = 0
        beta = 0.125
        for threshold in np.arange(0.5, 0.8, 0.03):
            y_val_pred = np.where(y_val_proba[:, 1] > threshold, 1, 0)

            validation_beta_score = fbeta_score(y_validation, y_val_pred, beta=beta)

            if validation_beta_score > best_fbeta_score:
                print(f'####improve Validation thres:{threshold} f-bete-{beta} score {validation_beta_score}')
                best_fbeta_score = validation_beta_score
                self.best_threshold = threshold

    def train_net(self, dataset_dict):
        self.compile_model(False)
        num_of_rows = self.network_params["Network_train_sample_size"]
        rows_index = np.random.choice(dataset_dict["X_train"].shape[0],
                                      size=num_of_rows,
                                      replace=False)

        self.model.fit(dataset_dict["X_train"][rows_index,:],
                  dataset_dict["y_train"][rows_index])

        y_val_proba = self.model.predict_proba(dataset_dict["X_validation"])
        self.update_best_threshold(y_val_proba,
                                   dataset_dict["y_validation"])


        y_train_pred = np.where(self.model.predict_proba(dataset_dict["X_train"][rows_index,:])[:, 1]
                                 > self.best_threshold, 1, 0)
        y_val_pred = np.where(y_val_proba[:, 1] > self.best_threshold, 1, 0)

        print('Train accuracy', accuracy_score(dataset_dict["y_train"][rows_index], y_train_pred))
        print('Validation accuracy', accuracy_score(dataset_dict["y_validation"], y_val_pred))

        print('Train precision', precision_score(dataset_dict["y_train"][rows_index], y_train_pred))
        print('Validation precision', precision_score(dataset_dict["y_validation"], y_val_pred))

        print('Train recall', recall_score(dataset_dict["y_train"][rows_index], y_train_pred))
        print('Validation recall', recall_score(dataset_dict["y_validation"], y_val_pred))

        print('Train f-beta score', fbeta_score(dataset_dict["y_train"][rows_index], y_train_pred, beta=0.25))
        validation_beta_score = fbeta_score(dataset_dict["y_validation"], y_val_pred, beta=0.25)
        print(f'Validation f-beta score {validation_beta_score}')

        return validation_beta_score

    def train_final_net(self, dataset_dict):
        print("train the best Network..")
        self.compile_model(bFinal=True)
        self.model.fit(dataset_dict["X_train"],
                       dataset_dict["y_train"])

        y_val_proba = self.model.predict_proba(dataset_dict["X_validation"])
        self.update_best_threshold(y_val_proba,
                                   dataset_dict["y_validation"])

        y_train_pred = np.where(self.model.predict_proba(dataset_dict["X_train"])[:, 1]
                                > self.best_threshold, 1, 0)
        y_val_pred = np.where(y_val_proba[:, 1] > self.best_threshold, 1, 0)

        print('Train accuracy', accuracy_score(dataset_dict["y_train"], y_train_pred))
        print('Validation accuracy', accuracy_score(dataset_dict["y_validation"], y_val_pred))

        print('Train precision', precision_score(dataset_dict["y_train"], y_train_pred))
        print('Validation precision', precision_score(dataset_dict["y_validation"], y_val_pred))

        print('Train recall', recall_score(dataset_dict["y_train"], y_train_pred))
        print('Validation recall', recall_score(dataset_dict["y_validation"], y_val_pred))

        print('Train f-beta score', fbeta_score(dataset_dict["y_train"], y_train_pred, beta=0.25))
        validation_beta_score = fbeta_score(dataset_dict["y_validation"], y_val_pred, beta=0.25)
        print(f'Validation f-beta score {validation_beta_score}')

        return validation_beta_score

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

        y_test_pred = np.where(self.model.predict_proba(ds_class["X_test"])[:, 1]
                               > self.best_threshold, 1, 0)
        np.savetxt(file_name, y_test_pred.astype(int), fmt='%i', delimiter='\n')
