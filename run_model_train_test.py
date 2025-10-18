# Importing necessary python packages
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import json
import sys
import os

# Compatibility shim
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.umath'] = np.core.umath
sys.modules['numpy._core.numeric'] = np.core.numeric

# Setting path and other input variable

MAIN_FOLDER = os.getcwd()
FOLDER_OUTPUT = "augmented_signals"
FOLDER_CONFIG = "config"
FOLDER_DATA = "datasets"
FOLDER_MODEL_OUTPUT = "model_results"
OPTIONS_TO_EVALUATE = ["ori", "aug"]
PATH_CONFIG_FILE = os.path.join(FOLDER_CONFIG,"config.json")



# Set the seed value for experiment reproducibility.
seed_value = 42

tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value) 





def main(args):


    import os
    import json

    # Opening config file
    dict_config = json.load(open(PATH_CONFIG_FILE, 'r'))

    # Opening datasets to be used in simulation
    with open(os.path.join(MAIN_FOLDER, FOLDER_DATA, args.use_case, dict_config[args.use_case]["dataset_proc_file"]), "rb") as file:
        dict_train_test_db = pickle.load(file)


    # Loading test and validation signals

    # Getting val partitions if available, otherwise
    # they will be generated given the provided val ratio
    if "X_val" in dict_train_test_db.keys():
        X_val = dict_train_test_db["X_val"]
    
    if "y_val" in dict_train_test_db.keys():
        y_val = dict_train_test_db["y_val"]

    X_test = dict_train_test_db["X_test"]
    y_test = dict_train_test_db["y_test"]



    # Opening original and augmented datasets
    with open(os.path.join(MAIN_FOLDER, FOLDER_OUTPUT, args.use_case + "_" + args.dataset_setup + ".pkl"), "rb") as file:
        dict_output = pickle.load(file)


    X_train = dict_output["X_train"]
    y_train = dict_output["y_train"]
    X_train_final = dict_output["X_train_final"]
    y_train_final = dict_output["y_train_final"]

    num_classes_db = len(set(dict_output["y_train"]))

    
    dict_model_results = {}
    dict_model_results["dataset_setup"] = dict_output

    if args.use_case == "mey_vibration":

        from tensorflow.keras.models import Sequential, load_model, Model
        from tensorflow.keras.layers import BatchNormalization,LeakyReLU,Dense,Dropout
        from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Flatten,ReLU
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow.keras.regularizers import l1_l2

        # Importing model architecture
        from model_setups import mey_vibration_model as chosen_model

        # Model params as defined by the authors
        n_conv_layers = 3
        use_batch_normalization = True
        n_dense_units = 128
        filter_size = 9
        learning_rate = 0.0001
        n_epochs = 100
        dropout_rate = 0.1
        n_dimension_0 = np.shape(X_train)[1]
        batch_size = 64

        
        for option_eval in OPTIONS_TO_EVALUATE:

            dict_model_results["history_"  + option_eval] = {}
            dict_model_results["test_loss_" + option_eval] = {}
            dict_model_results["test_accuracy_" + option_eval] = {}

            if option_eval == "ori":
                X_train_to_employ = X_train
                y_train_to_employ = y_train

            elif option_eval == "aug":
                X_train_to_employ = X_train_final
                y_train_to_employ = y_train_final

            else:
                raise ValueError(f"option_eval {option_eval} not recognized!")
            
            X_train_to_employ = np.reshape(X_train_to_employ, (X_train_to_employ.shape[0], X_train_to_employ.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

            # Train and test using original data
            for i_trial in range(args.n_trials):

                loaded_model = chosen_model(n_dimension_0, n_conv_layers, filter_size, use_batch_normalization, n_dense_units, dropout_rate)
                loaded_model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
                history = loaded_model.fit(X_train_to_employ, y_train_to_employ, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weights={0: 2.4981360671015844, 1: 0.625116604477612}, verbose=1)
        
                loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)

                dict_model_results["history_" + option_eval][f"trial_{i_trial}"] = history
                dict_model_results["test_loss_" + option_eval][f"trial_{i_trial}"] = loss
                dict_model_results["test_accuracy_" + option_eval][f"trial_{i_trial}"] = accuracy
            
        



    if args.use_case == "keras_ford":

        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dropout
        from tensorflow import keras

        # Importing model architecture
        from model_setups import keras_cnn_tf_from_scratch as chosen_model

        
        for option_eval in OPTIONS_TO_EVALUATE:

            dict_model_results["history_"  + option_eval] = {}
            dict_model_results["test_loss_" + option_eval] = {}
            dict_model_results["test_accuracy_" + option_eval] = {}

            if option_eval == "ori":
                X_train_to_employ = X_train
                y_train_to_employ = y_train

            elif option_eval == "aug":
                X_train_to_employ = X_train_final
                y_train_to_employ = y_train_final

            else:
                raise ValueError(f"option_eval {option_eval} not recognized!")

            
            for i_trial in range(args.n_trials):
                
                loaded_model = chosen_model(input_shape=(np.shape(X_train)[1], 1), num_classes=num_classes_db)
                loaded_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
                history = loaded_model.fit(X_train_to_employ,
                                y_train_to_employ,
                                epochs=5, # 500
                                validation_data=(X_val, y_val),
                                batch_size=32,
                                verbose=1)
            
                loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)

                dict_model_results["history_" + option_eval][f"trial_{i_trial}"] = history
                dict_model_results["test_loss_" + option_eval][f"trial_{i_trial}"] = loss
                dict_model_results["test_accuracy_" + option_eval][f"trial_{i_trial}"] = accuracy
            
            


    elif args.use_case == "magar_faultnet":


        import torch
        import torchvision
        import torch.nn as nn
        from torchvision.transforms import transforms
        from torch.utils.data import DataLoader
        from torch.optim import Adam
        from torch.autograd import Variable
        import torch.nn.functional as F
        import torch.utils.data as data_utils
        import torch.optim as optim
        from torch.utils.data import DataLoader, random_split

  
        # Importing model architecture
        from model_setups import magar_faultnet as chosen_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define original input params as in the paper/git
        NUMBER_EPOCHS = 5 # 100


        for option_eval in OPTIONS_TO_EVALUATE:

            dict_model_results["history_"  + option_eval] = {}
            dict_model_results["test_loss_" + option_eval] = {}
            dict_model_results["test_accuracy_" + option_eval] = {}

            if option_eval == "ori":
                X_train_to_employ = X_train
                y_train_to_employ = y_train

            elif option_eval == "aug":
                X_train_to_employ = X_train_final
                y_train_to_employ = y_train_final

            else:
                raise ValueError(f"option_eval {option_eval} not recognized!")
                
            loaded_model = chosen_model().double().to(device)

            # Performing data transformations as in the paper

            # Creating necessary functions

            def mean(data,no_elements):
                X=np.zeros((data.shape[0],data.shape[1]))
                for i in range(data.shape[1]-no_elements+1):
                    X[:,i]=np.mean(data[:,i:i+no_elements],axis=1)
                return X.astype(np.float16)

            def median(data,no_elements):
                X=np.zeros((data.shape[0],data.shape[1]))
                for i in range(data.shape[1]-no_elements+1):
                    X[:,i]=np.median(data[:,i:i+no_elements],axis=1)
                return X.astype(np.float16)

            def sig_image(data,size):
                X=np.zeros((data.shape[0],size,size))
                for i in range(data.shape[0]):
                    X[i]=(data[i,:].reshape(size,size))
                return X.astype(np.float16)
            
            

            # Computing rolling statistics

            SIG_SIZE = dict_train_test_db["y_test"]
            GROUP_CHAN_PARAM = dict_train_test_db["group_chan_param"]
            SIG_SIZE = dict_train_test_db["sig_size"]
            FS = dict_train_test_db["fs"]
            VAL_RATIO = dict_train_test_db["val_ratio"]
            WINDOW_STACKING_PARAM = dict_train_test_db["window_stacking_param"]

            # Resizing x train
            x = X_train_to_employ[:,:SIG_SIZE]
            channel_mean = (mean(x, GROUP_CHAN_PARAM)).astype(np.float16)
            x_m = sig_image(channel_mean, WINDOW_STACKING_PARAM)
            x_md = sig_image(x, WINDOW_STACKING_PARAM)
            x_n = sig_image(x, WINDOW_STACKING_PARAM)

            x_train = np.stack([x_n, x_m, x_md], axis=1).astype(np.float16)

            # Resizing x  test
            x = X_test[:,:SIG_SIZE]
            channel_mean = (mean(x, GROUP_CHAN_PARAM)).astype(np.float16)
            x_m = sig_image(channel_mean, WINDOW_STACKING_PARAM)
            x_md = sig_image(x, WINDOW_STACKING_PARAM)
            x_n = sig_image(x, WINDOW_STACKING_PARAM)

            x_test = np.stack([x_n, x_m, x_md], axis=1).astype(np.float16)

            del x


            for i_trial in range(args.n_trials):

                # Train, val and test original model
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)
                signals_train = torch.from_numpy(x_train).to(device)
                labels_train = torch.from_numpy(y_train_to_employ).to(device)
                signals_test = torch.from_numpy(x_test).to(device)
                labels_test = torch.from_numpy(y_test).to(device)

                train_tensor = data_utils.TensorDataset(signals_train, labels_train)
                train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=128, shuffle=True)

                # Creating validation data partition
                ratio_train = 1 - VAL_RATIO
                size_train = int(ratio_train*len(train_loader.dataset))
                size_val = len(train_loader.dataset) - size_train

                train_dataset, val_dataset = random_split(train_loader.dataset, [size_train, size_val])

                # Build data loaders for training and validation sets
                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=128)

                # Training and validation
                total_step = len(train_loader)

                list_loss = []
                list_acc = []

                # Training
                for epoch in range(NUMBER_EPOCHS):

                    loaded_model.train()
                    total_train_loss = 0.0

                    for i, (signals, labels) in enumerate(train_loader):

                        optimizer.zero_grad()
                        outputs = loaded_model(signals.double())
                        loss = criterion(outputs, labels.long())
                        list_loss.append(loss.item())

                        # Backpropagation and Adam optimization
                        loss.backward()
                        optimizer.step()

                        # Track the accuracy
                        total = labels.size(0)

                        _, predicted = torch.max(outputs.data, 1)
                        correct = (predicted == labels.long()).sum().item()
                        list_acc.append(correct/total)

                        if (epoch+1)%5==0 or epoch==0:
                            print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, Train Accuracy: {:.2f}%".format(epoch+1, NUMBER_EPOCHS, i+1, total_step, loss.item(), (correct/total)*100))


                # Validation
                loaded_model.eval()
                total_val_loss = 0.0
                total_correct = 0
                total_samples = 0

                with torch.no_grad():
                    for signals, labels in val_loader:

                        outputs = loaded_model(signals.double())
                        loss = criterion(outputs, labels.long())

                        total_val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_correct += (predicted == labels.long()).sum().item()
                        total_samples += labels.size(0)

                average_val_loss = total_val_loss/len(val_loader)
                validation_accuracy = total_correct/total_samples

                print("Validation loss: " + str(average_val_loss) + ", validation Accuracy: " + str(validation_accuracy))

                # Store model history
                history = {"model_params": sum(p.numel() for p in loaded_model.parameters()), 
                           "n_epochs": NUMBER_EPOCHS, 
                           "train_loss": list_loss, 
                           "train_accuracy": list_acc, 
                           "val_loss": average_val_loss, 
                           "val_accuracy": validation_accuracy}


                # Testing
                acc_list_test, loss_list_test = [], []

                test_tensor = data_utils.TensorDataset(signals_test, labels_test)
                test_loader = data_utils.DataLoader(dataset = test_tensor, batch_size=np.shape(signals_test)[0], shuffle=False)

                with torch.no_grad():

                    for i, (signals, labels) in enumerate(test_loader):

                        outputs = loaded_model(signals.double())
                        loss = criterion(outputs, labels.long())
                        loss_list_test.append(loss.item())

                        if epoch % 10 == 0:
                            print(loss)
                        
                        total = labels.size(0)

                        _, predict = torch.max(outputs.data, 1)
                        correct = (predict == labels.long()).sum().item()
                        acc_list_test.append(correct/total)

                        accuracy = np.nanmean(acc_list_test)
                        loss = np.nanmean(loss_list_test)

                        dict_test_results = {"eval_type": "classification", "metric": "accuracy", "value": accuracy, "loss": loss}

                dict_model_results["history_" + option_eval][f"trial_{i_trial}"] = history
                dict_model_results["test_loss_" + option_eval][f"trial_{i_trial}"] = loss
                dict_model_results["test_accuracy_" + option_eval][f"trial_{i_trial}"] = accuracy
                            

    elif args.use_case == "kws_ieeespl":



        import os
        import pathlib

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        import tensorflow as tf

        from tensorflow.keras import layers
        from tensorflow.keras import models
        from IPython import display

        # Importing model architecture
        from model_setups import kws_ieeespl_model as chosen_model
        
        # Importing other functions executed by the framework
        from model_setups import plot_spectrogram, get_spectrogram, make_spec_ds

        

        # Running training+validation and test steps 
        for option_eval in OPTIONS_TO_EVALUATE:

            dict_model_results["history_"  + option_eval] = {}
            dict_model_results["test_loss_" + option_eval] = {}
            dict_model_results["test_accuracy_" + option_eval] = {}

            if option_eval == "ori":
                X_train_to_employ = X_train
                y_train_to_employ = y_train

            elif option_eval == "aug":
                X_train_to_employ = X_train_final
                y_train_to_employ = y_train_final

            for i_trial in range(args.n_trials):

                # Transforming the training dataset, test and val splits back to tf datasets
                train_ds = tf.data.Dataset.from_tensor_slices((X_train_to_employ,
                                                           y_train_to_employ))
            
                train_ds = train_ds.shuffle(buffer_size=len(X_train_to_employ), seed=seed_value).batch(64).prefetch(tf.data.AUTOTUNE)
                train_ds.class_names = list(set(y_train_to_employ))
                label_names = np.array(train_ds.class_names)

                val_ds = tf.data.Dataset.from_tensor_slices((X_val,y_val))
                val_ds = val_ds.shuffle(buffer_size=len(X_val), seed=seed_value).batch(64).prefetch(tf.data.AUTOTUNE)
                val_ds.class_names = list(set(y_val))

                test_ds = tf.data.Dataset.from_tensor_slices((X_test,y_test))
                test_ds = test_ds.shuffle(buffer_size=len(X_test), seed=seed_value).batch(64).prefetch(tf.data.AUTOTUNE)
                test_ds.class_names = list(set(y_test))

                # Computing spectrogram datasets
                train_spectrogram_ds = make_spec_ds(train_ds)
                val_spectrogram_ds = make_spec_ds(val_ds)
                test_spectrogram_ds = make_spec_ds(test_ds)

                # Add Dataset.cache and Dataset.prefetch operations to reduce
                # read latency while training the model:

                train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
                val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
                test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE) 

                # Examine spectrograms for different examples in the data
                for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
                    break

                # Assuming you have these defined already:
                # example_spectrograms, label_names, train_spectrogram_ds

                input_shape = example_spectrograms.shape[1:]
                num_labels = len(label_names)

                loaded_model = chosen_model(train_spectrogram_ds, input_shape, num_labels)

                print(loaded_model.summary())

                loaded_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])

                history = loaded_model.fit(train_spectrogram_ds, validation_data=val_spectrogram_ds, epochs=5, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

                model_test_eval_return_dict = loaded_model.evaluate(test_spectrogram_ds, return_dict=True)

                dict_model_results["history_" + option_eval][f"trial_{i_trial}"] = history
                dict_model_results["test_loss_" + option_eval][f"trial_{i_trial}"] = model_test_eval_return_dict["loss"]
                dict_model_results["test_accuracy_" + option_eval][f"trial_{i_trial}"] = model_test_eval_return_dict["accuracy"]


    elif args.use_case == "keras_eeg":
    

        import pandas as pd
        import matplotlib.pyplot as plt
        import json
        import numpy as np
        import keras
        from keras import layers
        import tensorflow as tf
        from sklearn import preprocessing, model_selection
        import random   

        # Importing model architecture
        from model_setups import keras_eeg_model as chosen_model


        # Running training+validation and test steps 
        for option_eval in OPTIONS_TO_EVALUATE:

            dict_model_results["history_"  + option_eval] = {}
            dict_model_results["test_loss_" + option_eval] = {}
            dict_model_results["test_accuracy_" + option_eval] = {}

            if option_eval == "ori":
                X_train_to_employ = X_train
                y_train_to_employ = y_train

            elif option_eval == "aug":
                X_train_to_employ = X_train_final
                y_train_to_employ = y_train_final

            
            for i_trial in range(args.n_trials):

                num_classes = len(set(y_train_to_employ))

                X_train = np.asarray(X_train_to_employ).astype(np.float32).reshape(-1, 512, 1)
                y_train = np.asarray(y_train_to_employ).astype(np.float32).reshape(-1, 1)
                y_train = keras.utils.to_categorical(y_train)

                X_test = np.asarray(X_test).astype(np.float32).reshape(-1, 512, 1)
                y_test = np.asarray(y_test).astype(np.float32).reshape(-1, 1)
                y_test = keras.utils.to_categorical(y_test) 

                SHUFFLE_BUFFER_SIZE = dict_train_test_db["shuffle_buffer_size"]
                BATCH_SIZE = dict_train_test_db["batch_size"]

                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

                train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
                test_dataset = test_dataset.batch(BATCH_SIZE) 

                # Adding val split that is not in the Keras tutorial (in order to use
                # the same criterion for all methods/use cases)
                # Assume tf_train_dataset is already batched
                total_batches = len(train_dataset)
                val_size = int(float(dict_train_test_db["val_ratio"]) * total_batches)

                val_dataset = train_dataset.take(val_size)
                train_dataset = train_dataset.skip(val_size)


                # Label weights as defined for the eeg dataset in 
                # https://keras.io/examples/timeseries/eeg_signal_classification/

                weight_dict = {1: 0.9872413100261201, 
                            0: 0.975989551938919, 
                            14: 0.9841269841269842, 
                            13: 0.9061683745228049, 
                            9: 0.9838255977496484, 
                            8: 0.9059674502712477, 
                            11: 0.9847297568816556, 
                            10: 0.9063692987743621, 
                            18: 0.9838255977496484, 
                            17: 0.9057665260196905, 
                            16: 0.9373116335141651, 
                            15: 0.9065702230259193, 
                            2: 0.9211372312638135, 
                            12: 0.9525818766325096, 
                            3: 0.9245529435402853, 
                            4: 0.943841671689773, 
                            5: 0.9641350210970464, 
                            6: 0.981514968856741, 
                            7: 0.9443439823186659} 

                loaded_model = chosen_model(num_classes=num_classes) 

                print(loaded_model.summary())

                epochs = 30 # as defined in the Keras website

                callbacks = [
                    keras.callbacks.ModelCheckpoint(
                        "best_model.keras", save_best_only=True, monitor="loss"
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_top_k_categorical_accuracy",
                        factor=0.2,
                        patience=2,
                        min_lr=0.000001,
                    ),
                ]

                optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
                loss = keras.losses.CategoricalCrossentropy()


                loaded_model.compile(optimizer=optimizer,
                                   loss=loss,
                                   metrics=[
                                   keras.metrics.TopKCategoricalAccuracy(k=3),
                                   keras.metrics.AUC(),
                                   keras.metrics.Precision(),
                                   keras.metrics.Recall()])

                loaded_model_history = loaded_model.fit(train_dataset,
                                                        epochs=epochs,
                                                        callbacks=callbacks,
                                                        validation_data=val_dataset,
                                                        class_weight=weight_dict)
                
                
                

                loss, accuracy, auc, precision, recall = loaded_model.evaluate(test_dataset)

                dict_model_results["history_" + option_eval][f"trial_{i_trial}"] = loaded_model_history
                dict_model_results["test_loss_" + option_eval][f"trial_{i_trial}"] = loss
                dict_model_results["test_accuracy_" + option_eval][f"trial_{i_trial}"] = accuracy



    return dict_model_results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model trainer")

    parser.add_argument("--use_case", type=str, default = "keras_ford", required=False, 
                        help="Name of the model/dataset to be trained.")


    parser.add_argument("--dataset_setup", type=str, default = "2025_08_19_T_18_05_08", required=False, 
                        help="Name of the dataset to be augmented.")


    parser.add_argument("--n_trials", type=int, default = 1, required=False, 
                        help="Number of times to run full train and test runs.")

    args = parser.parse_args()

    
    dict_model_results = main(args)

    # Saving output dict (with original and augmented data)
    with open(os.path.join(MAIN_FOLDER, FOLDER_MODEL_OUTPUT,  "N_trials_" + str(args.n_trials) + "_" + args.use_case + "_" + args.dataset_setup + ".pkl"), "wb") as f:  
        pickle.dump(dict_model_results, f, protocol=3)



    print(dict_model_results)

