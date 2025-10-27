# Importing necessary python packages
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
import numpy as np
import itertools
import argparse
import random
import pickle
import torch
import json
import sys
import os

# Import project-specific functions and classes
from utils_modsurro import *


# Compatibility shim
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.umath'] = np.core.umath
sys.modules['numpy._core.numeric'] = np.core.numeric


# Setting path and other input variables

MAIN_FOLDER = os.getcwd()
FOLDER_CONFIG = "config"

# Opening config file
with open(os.path.join(FOLDER_CONFIG, "config.json"), "r", encoding="utf-8") as f:
    dict_config = json.load(f)

FOLDER_DATA = dict_config["general_params"]["folder_data"]
FOLDER_OUTPUT = dict_config["general_params"]["folder_output"]
FOLDER_PLOTS = dict_config["general_params"]["folder_plots"]






def main(args, dict_config):


    # Opening dataset to be used in simulation
    with open(os.path.join(MAIN_FOLDER, FOLDER_DATA, args.dataset_name, dict_config[args.dataset_name]["dataset_proc_file"]), "rb") as file:
        dict_train_test_db = pickle.load(file)


    # Loading original signals to be augmented
    X_train = dict_train_test_db["X_train"]
    y_train = dict_train_test_db["y_train"]
    fs = dict_train_test_db["fs"]

    if len(y_train) != np.shape(X_train)[0]:
    
        raise ValueError("y_train and X_train do not have the correct shape. The axis 0 of X_train should be the number of signals and match the lenght of y_train")


    # Instantiating generic data augmentation class given input data
    sig_augmenters = SignalAugmenters(X_train, 
                                      y_train, 
                                      n_replicates=args.n_replicates, 
                                      classes_to_augment=args.classes_to_aug, 
                                      fs=fs, 
                                      aug_mode=args.aug_mode, 
                                      frac_sigs_to_aug=args.frac_sigs_to_aug)

    

    # Running data augmentation function  

    if args.daug_method == "mod_surro":
        
        dict_output = sig_augmenters.perform_augmentation(aug_technique=args.daug_method,
                           frac_freq_shuffle=args.frac_freq_shuffle,
                           frac_sig_pairs_comp_coh=args.frac_sig_pairs_comp_coh)
        
        # Storing specific metadata for the chosen daug technique
        daug_tech_specs = {"frac_freq_shuffle": args.frac_freq_shuffle,
                           "frac_sig_pairs_comp_coh": args.frac_sig_pairs_comp_coh}
  
    elif args.daug_method == "tsaug":

        
        # Reading chosen TSAUG setup for data augmentation
        with open(os.path.join(FOLDER_CONFIG, args.tsaug_config + ".json"), "r", encoding="utf-8") as f:
            tsaug_config = json.load(f)

            
        dict_output = sig_augmenters.perform_augmentation(aug_technique=args.daug_method,
                           tsaug_config=tsaug_config)
        
        # Storing specific metadata for the chosen daug technique
        daug_tech_specs = {"tsaug_config": tsaug_config}

    elif args.daug_method == "seriesgan":

        # Reading chosen SERIESGAN setup for data augmentation
        with open(os.path.join(FOLDER_CONFIG, args.seriesgan_config + ".json"), "r", encoding="utf-8") as f:
            seriesgan_config = json.load(f)


        dict_output = sig_augmenters.perform_augmentation(aug_technique=args.daug_method,
                           seriesgan_config=seriesgan_config)
        
        # Storing specific metadata for the chosen daug technique
        daug_tech_specs = {"seriesgan_config": seriesgan_config}

    else:

        raise ValueError(f"Chosen data augmentation method '{args.daug_method}' not yet implemented or not known (maybe it was a typo?).")


    # Creating metadata entry with simulation info and params
    dict_output["metadata"] = {"method": args.daug_method,
                                "daug_tech_specs": daug_tech_specs,
                                "dataset_name": args.dataset_name,
                                "n_replicates": args.n_replicates,
                                "classes_to_aug": args.classes_to_aug,
                                "aug_mode": args.aug_mode,
                                "frac_sigs_to_aug": args.frac_sigs_to_aug,
                                "X_train": {"shape": np.shape(X_train)},
                                "y_train": {"shape": np.shape(y_train)},
                                "X_train_final": {"shape": np.shape(dict_output["X_train_final"])},
                                "y_train_final": {"shape": np.shape(dict_output["y_train_final"])},
                                "X_train_keep": {"shape": np.shape(dict_output["X_train_keep"])},
                                "array_ens_aug_data": {"shape": np.shape(dict_output["array_ens_aug_data"])},
                                "array_ens_aug_label": {"shape": np.shape(dict_output["array_ens_aug_label"])},
                                "effective_aug_ratio": np.shape(dict_output["X_train_final"])[0]/np.shape(X_train)[0]}
                                # effective_aug_ratio -> (size train db after daug)/(size train db originally (before daug))







    return dict_output



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run data augmentation.")


    parser.add_argument("--daug_method", type=str, default = "mod_surro", required=False, 
                        help="Name of the data augmentation method to implement.")
    
    parser.add_argument("--tsaug_config", type=str, default = "tsaug_setup_default", required=False, 
                        help="Name of the transformation/augmentation setup to execute if the daug method 'tsaug' is chosen in parameter --daug_method.")
    
    parser.add_argument("--seriesgan_config", type=str, default = "seriesgan_setup_default", required=False, 
                        help="Name of the transformation/augmentation setup to execute if the daug method 'seriesgan' is chosen in parameter --daug_method.")
    
    parser.add_argument("--dataset_name", type=str, default = "keras_ford", required=False, 
                        help="Name of the dataset to be augmented.")

    parser.add_argument("--n_replicates", type=int, default = 1, required=False, 
                        help="Num. of replicates each signal to augment will create.")
    
    parser.add_argument("--classes_to_aug", type=str, default="all", required=False, 
                        help="'minority' (aug only minority class), 'all' (aug all classes), or 'all_but_majority' (all classes but the most common one are aug.)")
    
    parser.add_argument("--frac_freq_shuffle", type=float, default= 0.5, required=False,
                        help="fraction of points of sorted_freq to shuffle. The higher the fraction the more freqs. will be shuffled. In the limit, if the fraction equals one, all frequencies are shuffled and the mod surro becomes the original surrogate.")

    parser.add_argument("--frac_sig_pairs_comp_coh", type=float, default=0.05, required=False,
                        help="Fraction of the total number of combination pairs to use to create the random sample of signal pairs to compute the spectral coherence. Depending on the number of signals to augment, the number of possible combination of pairs might be very large, so we need to consider a fraction.")
    
    parser.add_argument("--aug_mode", type=str, default="all", required=False,
                        help="augment all signals belonging to the class selected  to be augmented ('all') or pick a random sample of signals to augment ('random_sample')")
    
    parser.add_argument("--frac_sigs_to_aug", type=float, default=None, required=False,
                        help="fraction of the signals belonging to the class selected to augment if aug_mode == 'random_sample'")
   

    args = parser.parse_args()

    # Computing how long it takes to run a daug step
    start_time = datetime.now()

    dict_output = main(args, dict_config)

    end_time = datetime.now()

    time_to_run_daug = end_time - start_time
    time_to_run_daug_min = time_to_run_daug.total_seconds()/60


    # Create timestamp label string
    now = datetime.now()
    timestamp_str = now.strftime("%Y_%m_%d_T_%H_%M_%S")
    current_simulation_id = args.dataset_name + "_" + timestamp_str

    # Update metadata field with similation id
    dict_output["metadata"]["simulation_id"] = current_simulation_id
    dict_output["metadata"]["daug_time_in_min"] = time_to_run_daug_min


    # Saving output dict (with original and augmented data)
    with open(os.path.join(MAIN_FOLDER, FOLDER_OUTPUT, current_simulation_id + ".pkl"), "wb") as f:  
        pickle.dump(dict_output, f)

        
    # Plotting examples of original vs. augmented signal for one sample of each class
    augmented_classes = list(set(dict_output["array_ens_aug_label"]))


    with open(os.path.join(MAIN_FOLDER, FOLDER_DATA, args.dataset_name, dict_config[args.dataset_name]["dataset_proc_file"]), "rb") as file:
        dict_train_test_db = pickle.load(file)

    X_train = dict_train_test_db["X_train"]
    y_train = dict_train_test_db["y_train"]

    # Save the plot to a PDF file
    os.makedirs(os.path.join(MAIN_FOLDER, FOLDER_OUTPUT, FOLDER_PLOTS, current_simulation_id), exist_ok=True)
        

    for my_class in augmented_classes:


        idx_aug_class = [i for i, v in enumerate(dict_output["array_ens_aug_label"] == my_class) if v]
        idx_ori_class = [i for i, v in enumerate(y_train == my_class) if v]

        # Randomly picking one signal sample from current class to plot
        idx_aug_sig = random.choice(idx_aug_class)
        idx_ori_sig = random.choice(idx_ori_class)

        aug_sig_sample = dict_output["array_ens_aug_data"][idx_aug_sig]
        ori_sig_sample = X_train[idx_ori_sig]

        # Plot
        plt.plot(aug_sig_sample, color="black", label="Aug. signal")
        plt.plot(ori_sig_sample, color="gray", label="Ori. signal")
        plt.xlabel("time (n)")
        plt.ylabel("Values")
        
        if type(my_class) is str:
            plt.title(f"Example aug. vs. original signal of class '{my_class}'")
        else:
            plt.title(f"Example aug. vs. original signal of class '{int(my_class)}'")

        plt.legend(loc="upper right")
        plt.savefig(os.path.join(MAIN_FOLDER, FOLDER_OUTPUT, FOLDER_PLOTS, current_simulation_id, "plot" + "_" + current_simulation_id + "_" + "class_" + str(int(my_class)) + ".pdf")
)   
        plt.close()

        
    print(dict_output)

