from __future__ import print_function
import numpy as np
from My_devol import MyDEvol, MyGenomeHandler
from My_devol.my_genome_handler import fbeta_keras
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

def get_best_threshold(y_val_proba, y_validation, y_train_proba, y_train):
    best_threshold = 0.5
    best_fbeta_score_valid = 0
    best_fbeta_score_train = 0
    beta = 0.25
    for threshold in np.arange(0.5, 0.8, 0.001):
        y_val_pred = np.where(y_val_proba[:, 0] > threshold, 1, 0)
        y_train_pred = np.where(y_train_proba[:, 0] > threshold, 1, 0)

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

def devol_train_final_model(model, dataset_dict,EarlyStopping_patience=6):
    str_header = "#"*80
    str_header2 = "$" * 80
    print(str_header)
    print(f"best Network.. train_final_net  with param TODO:")
    print(str_header)

    fit_params = {
        'x': dataset_dict['X_train'],
        'y': dataset_dict['y_train'],
        'validation_split': 0.1,
        'epochs': 300,
        'verbose': 1,
        'validation_data':(dataset_dict['X_validation'],
                           dataset_dict['y_validation']),
        'callbacks': [
            EarlyStopping(monitor='val_loss',
                          patience=EarlyStopping_patience,
                          verbose=0)
        ]
    }

    model.fit(**fit_params)

    y_val_proba = model.predict_proba(dataset_dict["X_validation"])
    y_train_proba = model.predict_proba(dataset_dict["X_train"])

    y_train_pred = np.where(y_train_proba[:, 0]
                            > 0.5, 1, 0)
    y_val_pred = np.where(y_val_proba[:, 0] > 0.5, 1, 0)

    print(str_header)
    print("Before improve threshold")
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

    # Improve Threshold!!!
    best_threshold = get_best_threshold(y_val_proba,
                                        dataset_dict["y_validation"],
                                        y_train_proba,
                                        dataset_dict["y_train"])

    y_train_pred = np.where(y_train_proba[:, 0]
                            > best_threshold, 1, 0)
    y_val_pred = np.where(y_val_proba[:, 0] > best_threshold, 1, 0)

    print(str_header2)
    print(f"After improve threshold to {best_threshold}")
    print(str_header2)
    print('Train accuracy', accuracy_score(dataset_dict["y_train"], y_train_pred))
    print('Validation accuracy', accuracy_score(dataset_dict["y_validation"], y_val_pred))

    print('Train precision', precision_score(dataset_dict["y_train"], y_train_pred))
    print('Validation precision', precision_score(dataset_dict["y_validation"], y_val_pred))

    print('Train recall', recall_score(dataset_dict["y_train"], y_train_pred))
    print('Validation recall', recall_score(dataset_dict["y_validation"], y_val_pred))

    print('Train f-beta score', fbeta_score(dataset_dict["y_train"], y_train_pred, beta=0.25))
    validation_beta_score = fbeta_score(dataset_dict["y_validation"], y_val_pred, beta=0.25)
    print(f'Validation f-beta score {validation_beta_score}')
    print(str_header2)
    print(str_header2)
    return model, best_threshold

def WriteResToFile(model,best_threshold, ds_class,file_name):
    print(f"Write tests results to File {file_name}..")

    y_test_pred = np.where(model.predict_proba(ds_class["X_test"])[:, 0]
                           > best_threshold, 1, 0)
    np.savetxt(file_name, y_test_pred.astype(int), fmt='%i', delimiter='\n')

def DevolTrainExistModel(dataset_dict,MODEL_NAME,FILE_NAME,EarlyStopping_patience):
    ge = "@"*80

    suffix = f"#pati-{EarlyStopping_patience}"
    new_MODEL_NAME = MODEL_NAME.replace(".h5", f"{suffix}.h5")
    new_FILE_NAME = FILE_NAME.replace(".txt", f"{suffix}.txt")

    print(ge)
    print(ge)
    print(f"Run over exist's model {new_MODEL_NAME}")
    print(ge)

    split_dim = dataset_dict['X_train'].shape[1] / 4
    dataset_dict['X_train'] = np.stack(np.split(dataset_dict['X_train'], split_dim , 1), 2)
    dataset_dict['X_validation'] = np.stack(np.split(dataset_dict['X_validation'], split_dim, 1), 2)
    dataset_dict['X_test'] = np.stack(np.split(dataset_dict['X_test'], split_dim, 1), 2)

    print("Loading model {MODEL_NAME}")
    model = load_model(MODEL_NAME, custom_objects={"fbeta_keras": fbeta_keras})
    print(model.summary())
    model,best_t = devol_train_final_model(model,dataset_dict)
    WriteResToFile(model,best_t,dataset_dict,new_FILE_NAME)
    model.save(new_MODEL_NAME)


def DevolMain(dataset_dict,generations,population,MODEL_NAME,FILE_NAME):
    # TODO: Delete after stableize
    generations=1
    population=4


    num_of_s = 300000
    # dataset_dict['X_train'] = dataset_dict['X_train'][:num_of_s, :]
    # dataset_dict['y_train'] = dataset_dict['y_train'][:num_of_s, :]
    # dataset_dict['X_validation'] = dataset_dict['X_validation'][:num_of_s, :]
    # dataset_dict['y_validation'] = dataset_dict['y_validation'][:num_of_s, :]
    # dataset_dict['X_test'] = dataset_dict['X_test'][:num_of_s, :]
    ## TODO: until this part..

    split_dim = dataset_dict['X_train'].shape[1] / 4

    dataset_dict['X_train'] = np.stack(np.split(dataset_dict['X_train'], split_dim , 1), 2)
    dataset_dict['X_validation'] = np.stack(np.split(dataset_dict['X_validation'], split_dim, 1), 2)
    dataset_dict['X_test'] = np.stack(np.split(dataset_dict['X_test'], split_dim, 1), 2)

    dataset = ((dataset_dict['X_train'][:num_of_s, :],
                dataset_dict['y_train'][:num_of_s, :]),
               (dataset_dict['X_validation'][:num_of_s, :],
                dataset_dict['y_validation'][:num_of_s, :]))
    s = dataset_dict['X_train'].shape

    genome_handler = MyGenomeHandler(max_conv_layers=2,
                                     max_dense_layers=2,  # includes final dense layer
                                     max_filters=10,
                                     max_dense_nodes=24,
                                     input_shape=s[1:],
                                     dropout=True)
    epochs = 6
    devol = MyDEvol(genome_handler)
    model = devol.run(dataset=dataset,
                      num_generations=generations,
                      pop_size=population,
                      epochs=epochs)


    print(model.summary())
    model,best_t = devol_train_final_model(model,dataset_dict)
    WriteResToFile(model,best_t,dataset_dict,FILE_NAME)
    model.save(MODEL_NAME)
