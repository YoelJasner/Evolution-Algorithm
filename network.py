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
        self.best_threshold = 0.5

    def compile_model(self,bFinal=False):
        # Get our network parameters.
        max_iter = self.network_params['final_max_iter'] if bFinal else self.network_params['max_iter']
        n_iter_no_change = 10 if bFinal else 10
        self.best_threshold = 0.5
        self.model = MLPClassifier(max_iter=max_iter,
                                   verbose=2,
                                   n_iter_no_change=n_iter_no_change,
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
        logging.info("Network threshold: %.2f%%" % (self.best_threshold))
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

    def update_best_threshold(self, y_val_proba, y_validation,y_train_proba, y_train):
        self.best_threshold = 0.5
        best_fbeta_score_valid = 0
        best_fbeta_score_train = 0
        beta = 0.25
        for threshold in np.arange(0.5, 0.8, 0.0025):
            y_val_pred = np.where(y_val_proba[:, 1] > threshold, 1, 0)
            y_train_pred = np.where(y_train_proba[:, 1] > threshold, 1, 0)

            curr_validation_beta_score = fbeta_score(y_validation, y_val_pred, beta=beta)
            curr_train_beta_score = fbeta_score(y_train, y_train_pred, beta=beta)

            if curr_validation_beta_score >= best_fbeta_score_valid:
            #if curr_train_beta_score >= best_fbeta_score_train:
                # and curr_train_beta_score >= best_fbeta_score_train:

                best_fbeta_score_valid = curr_validation_beta_score
                best_fbeta_score_train = curr_train_beta_score
                self.best_threshold = threshold

        header_note = "#" * 80
        print(header_note)
        print(f'#### improve thres:{self.best_threshold} With:')
        print(f'validation f-beta-{beta} score {best_fbeta_score_valid}')
        print(f'train f-beta-{beta} score {best_fbeta_score_train}')
        print(header_note)

    def train_net(self, dataset_dict):
        self.compile_model(False)
        num_of_rows = self.network_params["Network_train_sample_size"]
        rows_index = np.random.choice(dataset_dict["X_train"].shape[0],
                                      size=num_of_rows,
                                      replace=False)
        print(f"train_net with param{self.network_params}")
        self.model.fit(dataset_dict["X_train"][rows_index,:],
                  dataset_dict["y_train"][rows_index])

        y_val_proba = self.model.predict_proba(dataset_dict["X_validation"])
        y_train_proba = self.model.predict_proba(dataset_dict["X_train"][rows_index,:])
        self.update_best_threshold(y_val_proba,
                                   dataset_dict["y_validation"],
                                   y_train_proba,
                                   dataset_dict["y_train"][rows_index])


        y_train_pred = np.where(y_train_proba[:, 1]
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
        str_header = "#"*80
        print(str_header)
        print(f"best Network.. train_final_net  with param{self.network_params}")
        print(str_header)
        self.compile_model(bFinal=True)
        self.model.fit(dataset_dict["X_train"],
                       dataset_dict["y_train"])

        y_val_proba = self.model.predict_proba(dataset_dict["X_validation"])
        y_train_proba = self.model.predict_proba(dataset_dict["X_train"])
        self.update_best_threshold(y_val_proba,
                                   dataset_dict["y_validation"],
                                   y_train_proba,
                                   dataset_dict["y_train"])

        y_train_pred = np.where(y_train_proba[:, 1]
                                > self.best_threshold, 1, 0)
        y_val_pred = np.where(y_val_proba[:, 1] > self.best_threshold, 1, 0)
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
        self.accuracy = validation_beta_score
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
