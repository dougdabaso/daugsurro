# Packages to run seriesgan daug
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import logging
tf.get_logger().setLevel(logging.ERROR)

# 1. SeriesGan model
from SeriesGAN.seriesgan import seriesgan

# 2. Data loading
from SeriesGAN.data_loading import real_data_loading, sine_data_generation

# 3. Metrics
from SeriesGAN.metrics.discriminative_metrics import discriminative_score_metrics
from SeriesGAN.metrics.predictive_metrics import predictive_score_metrics
from SeriesGAN.metrics.visualization_metrics import visualization

# Packages to run tsaug daug
import tsaug

# Packages to run mod surro method

from scipy import signal
import numpy as np
import itertools
import random
import json
import os




# Setting path and other input variables

MAIN_FOLDER = os.getcwd()
FOLDER_CONFIG = "config"

# Opening config file
with open(os.path.join(FOLDER_CONFIG, "config.json"), "r", encoding="utf-8") as f:
    dict_config = json.load(f)



MIN_NUM_TUPLES = int(dict_config["general_params"]["min_num_tuples"])
EPS_TO_HALT_MAG_COH_COMP = float(dict_config["general_params"]["eps_to_halt_mag_coh_comp"])
N_POINTS_MAG_COH_CONV_HAT = int(dict_config["general_params"]["n_points_mag_coh_conv_hat"])
MAX_COH_CURVES = int(dict_config["general_params"]["max_coh_curves"])



class SignalAugmenters:
  """
  Stores signal augmenter functions implementing different methods
  (modified surrogate, tsaug, gan-based augmentation, etc.)
  """

  def __init__(self,
               X_train, 
               y_train, 
               n_replicates=1, 
               classes_to_augment="all", 
               fs=1, 
               aug_mode="all", 
               frac_sigs_to_aug=0.5):
      
      """
      Initialize SignalAugmenters class.

      Args:

      X_train: Training signals to augment.
        
      y_train: Labels of the training signals.
        
      fs: The sampling frequency to consider (if any).
        
      n_replicates: number of modified surrogates to create for
      each signal.
        
      classes_to_augment: which classes (labels) of the parsed
      training dataset we should augment, options are 'all',
      (all classes), 'minority' (only the minority class), and
      'all_but_majority' (all classes except of the most common)

      aug_mode: augment all signals belonging to the class selected 
      to be augmented ('all') or pick a random sample of signals
      to augment ('random_sample')
      
      frac_sigs_to_aug: fraction of the signals belonging to the 
      class selected to augment if aug_mode == 'random_sample'


      """
      self.X_train = X_train
      self.y_train = y_train
      self.n_replicates = n_replicates
      self.classes_to_augment = classes_to_augment
      self.fs = fs
      self.aug_mode = aug_mode
      self.frac_sigs_to_aug = frac_sigs_to_aug


      # Identifying classe to augment

      if classes_to_augment == "all":

        self.actual_classes_to_aug = list(set(y_train))

      elif classes_to_augment == "minority":

        self.actual_classes_to_aug = min(list(y_train), key=list(y_train).count)

      elif classes_to_augment == "all_but_majority":

        self.actual_classes_to_aug = list(set([y_t for y_t in y_train if y_t != max(y, key=list(y_train).count)]))

      else:

        raise ValueError("Parameter classes_to_augment with invalid value (possibilities: 'all', 'minority', 'all_but_majority')")

        
      # Augmenting signals

      # Create boolean mask to pick signals to augment
      mask_in = np.isin(y_train, self.actual_classes_to_aug)


      # Use the mask to split the arrays
      self.y_train_aug = y_train[mask_in] # these signals will be augmented
      self.X_train_aug = X_train[mask_in]

      self.y_train_keep = y_train[~mask_in] # these signals will not be augmented
      self.X_train_keep = X_train[~mask_in]

      # Setting up aug mode ('all' or 'random_sample')
      # Augment a random sample of the signals from the class 
      # selected to augment

      if aug_mode == 'random_sample':
      
        N_sigs_to_aug = int(frac_sigs_to_aug*np.shape(self.X_train_aug)[0])
        sample_indices = np.random.choice(len(self.X_train_aug), size=N_sigs_to_aug, replace=False)

        self.X_train_aug = self.X_train_aug[sample_indices]
        self.y_train_aug = self.y_train_aug[sample_indices]




  def perform_augmentation(self,
                           aug_technique="mod_surro",
                           frac_freq_shuffle=0.25,
                           frac_sig_pairs_comp_coh=0.10,
                           tsaug_config={},
                           seriesgan_config={},
                           renyi_mask_config={}):
     
    """
    Takes as input the chosen method and performs data
    augmentation after instantiating SignalAugmenters class.

    Optional Args:

    (aug_technique = 'mod_surro')

    frac_freq_shuffle: fraction of points of sorted_freq to shuffle.
    The higher the fraction the more freqs. will be shuffled.
    In the limit, if the fraction equals one, all frequencies are
    shuffled and the mod surro becomes the original surrogate.

    frac_sig_pairs_comp_coh: Fraction of the total number of combination pairs to use 
    to create the random sample of signal pairs to compute the spectral coherence.
    Depending on the number of signals to augment, the number of possible
    combination of pairs might be very large, so we need to consider a fraction.

    (aug_technique = 'tsaug')

    tsaug_config (dict): dictionary with the setup to be used for daug
    with tsaug (i.e., with the transformations to use and their params)


    Returns:

    dict_output: Output dict containing the following fields:
    
      - X_train_keep: Set of signals that have not been augmented (e.g., from
    the majority class). Depending on the augmentation option, this set
    of signal can be empty (e.g., if we choose to augment all signals).

      - y_train_keep: Label of set of signals that have not been augmented
    
      - array_ens_aug_label: Set of augmented signals
    
      - array_ens_aug_label: Labels of augmented signals

      - X_train_final: Set of augmented plus original signals

      - y_train_final: Set of augmented plus original signal labels

    """

    # Reading input params and data

    X_train = self.X_train
    y_train = self.y_train
    fs = self.fs
    n_replicates = self.n_replicates
    aug_mode = self.aug_mode
    frac_sigs_to_aug = self.frac_sigs_to_aug
    y_train_aug = self.y_train_aug
    y_train_keep = self.y_train_keep
    X_train_aug = self.X_train_aug
    X_train_keep = self.X_train_keep

    # Initialize dict to contain aug_method_metadata
    aug_method_metadata = {}


    # Performing some pre-calculations for the implemented
    # daug methods that require to do so  

    # Pre-calculations of mod_surro method
    if aug_technique == "mod_surro":
       
      # Computing spectral coherence of signals to augment

      available_classes = list(set(y_train_aug))

      dict_important_bands = {}
      for current_class in available_classes:
        
        # Create boolean mask to pick signals to augment
        mask_in_current_class = np.isin(y_train_aug, current_class)
        y_train_aug_current_class = y_train_aug[mask_in_current_class]
        X_train_aug_current_class = X_train_aug[mask_in_current_class]

        mag_coherence_list, _, freq_points, _, list_avg_error = computing_coherence(len(y_train_aug_current_class),
                                                                                    X_train_aug_current_class,
                                                                                    fs, 
                                                                                    frac_sig_pairs_comp_coh=frac_sig_pairs_comp_coh)
      
        # Getting the freq indexes corresponding to the largest values
        # of mag spectral coherence, which are assumed to be the most
        # relevant frequencies of the augmented classes

        sorted_indices = np.argsort(np.mean(mag_coherence_list,axis=0))[::-1]
        freq_points = np.array(freq_points)
        sorted_freq = freq_points[sorted_indices]

        dict_important_bands[current_class] = {"mag_coherence_list": mag_coherence_list,
                                             "freq_points": freq_points,
                                             "sorted_indices": sorted_indices,
                                             "sorted_freq": sorted_freq,
                                             "list_avg_error": list_avg_error,
                                             "y_train_aug_current_class": y_train_aug_current_class,
                                             "X_train_aug_current_class": X_train_aug_current_class}
      

      aug_method_metadata[aug_technique] = {"dict_important_bands": dict_important_bands}
      

    # Pre-calculations of tsaug method
    elif aug_technique == "tsaug":
       

        # Instantiating tsaug augmenter

        my_daug_std = np.nanstd(X_train_aug)
        signal_len = np.shape(X_train_aug)[0]

        if len(tsaug_config) > 0:
          tsaug_obj = TsaugAugmenter(my_daug_std, signal_len, tsaug_config)
        else:
           raise ValueError("tsaug_config dictionary cannot be empty for augmentation technique 'tsaug'.")
        
        my_augmenter = tsaug_obj.create_aug_obj()
        aug_method_metadata[aug_technique] = {"my_daug_std": my_daug_std,
                                              "tsaug_obj_augmenter": tsaug_obj}



    # Performing data augmentation itself
    N_sigs_to_aug = np.shape(X_train_aug)[0]

    # For mod_surr and tsaug, each signal will be augmented
    # individually. For the seriesgan method, we will augment
    # all signals belonging to each class chosen to be augmented 

    if aug_technique == "seriesgan":
        
      available_classes_to_aug = list(set(y_train_aug))

      for i_class in range(len(available_classes_to_aug)):
        
        class_to_aug = available_classes_to_aug[i_class]
        mask_aug_class = np.isin(y_train_aug, class_to_aug)
        X_train_aug_current_class = X_train_aug[mask_aug_class]
        y_train_aug_current_class = y_train_aug[mask_aug_class]

        n_sigs_to_create = int(n_replicates*np.shape(X_train_aug_current_class)[0])

        array_aug_label = np.tile(y_train_aug_current_class, n_replicates)

        array_aug_data = seriesgan(X_train_aug_current_class.reshape(np.shape(X_train_aug_current_class)[0],
                                                                      np.shape(X_train_aug_current_class)[1],
                                                                      1), seriesgan_config, n_sigs_to_create)
        print(np.shape(array_aug_data))

        array_aug_data = array_aug_data.reshape(np.shape(array_aug_data)[0], np.shape(array_aug_data)[1])

        if i_class == 0:
          array_ens_aug_data = array_aug_data
          array_ens_aug_label = array_aug_label
        else:
          array_ens_aug_data = np.concatenate([array_aug_data, array_ens_aug_data], axis=0)
          array_ens_aug_label = np.concatenate([array_aug_label, array_ens_aug_label], axis=0)


        filename_prov_seriesgan = "prov_seriesgan_" + str(i_class) + ".pkl"

        dict_prov_seriesgan_data = {"array_ens_aug_data": array_ens_aug_data, 
                                    "array_ens_aug_label": array_ens_aug_label}
        
        
        try:
          import pickle
          with open(filename_prov_seriesgan, "wb") as f:  
            pickle.dump(dict_prov_seriesgan_data, f, protocol=3)
          
          # Sending to drive right away
          import shutil
          DEST_DRIVE_FOLDER="/content/drive/MyDrive/daugsurro/"
          SRC_COLAB_FILEPATH = filename_prov_seriesgan
          shutil.copy(SRC_COLAB_FILEPATH, DEST_DRIVE_FOLDER)

        except:
          print("Copying to drive directly did not work out.")
          print("The DEST_DRIVE_FOLDER variable was " + DEST_DRIVE_FOLDER)
          print("The SRC_COLAB_FILEPATH variable was " + SRC_COLAB_FILEPATH)
          print("The command used was shutil.copy(SRC_COLAB_FILEPATH, DEST_DRIVE_FOLDER)")
        


    else: # methods where we augment each signal individually

      for i_sig in range(N_sigs_to_aug):

        input_signal = X_train_aug[i_sig]
        input_label = y_train_aug[i_sig]

        print(i_sig)

        if aug_technique == "mod_surro": 

          array_aug_data = compute_mod_surro(input_signal, 
                                             fs, 
                                             dict_important_bands[input_label]["sorted_freq"],
                                             frac_freq_shuffle, 
                                             n_replicates)
          

          array_aug_label = input_label*np.ones(n_replicates)

          print(str(i_sig) + " augmented.")

          if i_sig == 0:
            array_ens_aug_data = array_aug_data
            array_ens_aug_label = array_aug_label
          else:
            array_ens_aug_data = np.concatenate([array_aug_data, array_ens_aug_data], axis=0)
            array_ens_aug_label = np.concatenate([array_aug_label, array_ens_aug_label], axis=0)


        elif aug_technique == "tsaug":
          
          if i_sig == 0:
            list_ens_aug_label = []

          list_aug_label = []
          for i_rep in range(n_replicates):

            aug_data = my_augmenter.augment(input_signal)
            aug_data = aug_data.reshape(1,np.shape(aug_data)[0]) 
            aug_label = input_label  

            if i_rep == 0:
              array_aug_data = aug_data
            else:
              array_aug_data = np.concatenate([array_aug_data, aug_data], axis=0)
            
            list_aug_label.append(aug_label)

          if i_sig == 0:
            array_ens_aug_data = array_aug_data
          else:
            array_ens_aug_data = np.concatenate([array_aug_data, array_ens_aug_data], axis=0)
          
          list_ens_aug_label += list_aug_label
          array_ens_aug_label = np.array(list_ens_aug_label)
 

    # Combining back array_ens_aug_data and array_ens_aug_label to not augmented conterparts
    
                                   # part not sent to augmentation, part sent to augmentation, part augmented
    X_train_final = np.concatenate([X_train_keep, X_train_aug, array_ens_aug_data], axis=0)
    y_train_final = np.concatenate([y_train_keep, y_train_aug, array_ens_aug_label], axis=0)




    dict_output = {"X_train": X_train,
                  "y_train": y_train,
                  "X_train_final": X_train_final,
                  "y_train_final": y_train_final,
                  "X_train_keep": X_train_keep,
                  "y_train_keep": y_train_keep,
                  "array_ens_aug_label": array_ens_aug_label,
                  "array_ens_aug_data": array_ens_aug_data,
                  "aug_method_metadata": aug_method_metadata} 


    return dict_output




class TsaugAugmenter:
  """Executes time series augmentation utilizing the TSAUG library
    and the default augmentation examples given in the initual tutorial.
    TSAUG time series augmentation: https://tsaug.readthedocs.io/""" 

  def __init__(self, my_daug_std, signal_len, tsaug_config):
    """
    Initialize TsaugAugmenter with parameters dependent on the original
    dataset to be augmented.

    Args:

    my_daug_std (float): Estimated standard deviation of the signals
    signal_len (int): Length of the input signals
    tsaug_config (dict): dictionary with the setup to be used for daug
    with tsaug (i.e., with the transformations to use and their params)
    """

    # Making sure numerical variables are pure float and it (not numpy's)
    # otherwise tsaug function will crash

    self.my_daug_std = float(my_daug_std)
    self.signal_len = int(signal_len)

    self.tsaug_config = tsaug_config


  def create_aug_obj(self):
    """
    Creates the augmenter object based on the provided augmentation
    setup (tsaug_config), as well as metrics from the data (len and std).

    Returns: augmenter object with .augment() method to create new data.
    """      

    tsaug_config = self.tsaug_config

    my_daug_std = float(self.my_daug_std)
    signal_len = int(self.signal_len)

    available_transf = list(tsaug_config.keys())

    for i_transf in range(len(available_transf)):

      transf = available_transf[i_transf]

      if transf == "AddNoise":
          my_current_augmenter = tsaug.AddNoise(scale=(my_daug_std*(1 - tsaug_config["AddNoise"]["scale"]), my_daug_std*(1 + tsaug_config["AddNoise"]["scale"])))

      elif transf == "Dropout":
          my_current_augmenter = tsaug.Dropout(p=tsaug_config["Dropout"]["p"], size=(tsaug_config["Dropout"]["size"][0], tsaug_config["Dropout"]["size"][1]))

      elif transf == "Drift":
          my_current_augmenter = tsaug.Drift(max_drift=tsaug_config["Drift"]["max_drift"], n_drift_points=tsaug_config["Drift"]["n_drift_points"])

      elif transf == "Pool":
          my_current_augmenter = tsaug.Pool(size=tsaug_config["Pool"]["size"])

      elif transf == "Convolve":
          my_current_augmenter = tsaug.Convolve(window=tsaug_config["Convolve"]["window"], size=int(tsaug_config["Convolve"]["frac_size"]*signal_len))

      elif transf == "Quantize":
          my_current_augmenter = tsaug.Quantize(n_levels=tsaug_config["Quantize"]["n_levels"])

      elif transf == "TimeWarp":
          my_current_augmenter = tsaug.TimeWarp(n_speed_change=tsaug_config["TimeWarp"]["n_speed_change"], max_speed_ratio=tsaug_config["TimeWarp"]["max_speed_ratio"])


      if i_transf == 0:
          my_augmenter = my_current_augmenter

      else:
          my_augmenter = my_augmenter + my_current_augmenter

    return my_augmenter
   

    





def find_closest_freq_value(freq_values, target_freq, dist_function="abs_diff"):
  """
  Finds the closest frequency value in a given list to a target frequency.

  Args:
    freq_values: A numpy array representing the list of frequency values.

    target_freq: A float representing the target frequency.

    dist_function: A string representing the distance function to use.
      Can be either "abs_diff" or "euclidean".

  Returns:
    closest_freq_value: The closest frequency value to the target frequency.
    id_argmin: The index of closest_freq_value inside freq_values
  """

  if dist_function == "abs_diff":

    id_argmin = np.argmin([np.abs(x - target_freq) for x in freq_values])
    closest_freq_value = freq_values[id_argmin]

  elif dist_function == "euclidean":
    id_argmin = np.argmin([np.linalg.norm(x - target_freq) for x in freq_values])
    closest_freq_value = freq_values[id_argmin]

  else:
    raise ValueError("Invalid distance function. Must be either 'abs_diff' or 'euclidean'.")

  return closest_freq_value, int(id_argmin)


def computing_coherence(number_of_coh_spec_to_compute, signals_to_augment, fs, frac_sig_pairs_comp_coh=None):
    """
    Computes spectral conherence for a random sample of signal pairs from the
    collection of signals to augment. The number of sample pairs to randomly
    create is defined by the users. The outputs of this function allow to define
    frequency points/bands for which the magnitude coherence values are the lowest.
    
    Args:
    number_of_coh_specs_to_compute: An int representing the number of pairs of signals
    (which will be used to compute one coherence spectrum) to consider
    
    signals_to_augment: A list storing the signals (arrays) to augment
    
    fs: The sampling frequency of the signals
    
    frac_sig_pairs_comp_coh: Fraction of the total number of combination pairs to use 
    to create the random sample of signal pairs to compute the spectral coherence.
    Depending on the number of signals to augment, the number of possible
    combination of pairs might be very large, so we need to consider a fraction.
    
    Returns:
    mag_coherence_list: List with the mag coherence spectra computed for each
    pair of signals randomly smapled.
    
    phase_coherence_list: List with the phase coherence spectra computed for each
    pair of signals randomly smapled.
    
    freq_points: Array with frequency points of mag and phase coherence arrays.
    
    signal_matches_coherence: Pairs of signals that were randomly sampled.
    
    """

    mag_coherence_list, phase_coherence_list, freq_list = [], [], []

    # Compute list of signal tuples to compute coherence
    signal_matches_coherence = list(itertools.combinations(range(number_of_coh_spec_to_compute-1), 2))

    if frac_sig_pairs_comp_coh is not None:
        
      N_comb_to_use = int(len(signal_matches_coherence)*frac_sig_pairs_comp_coh)
      signal_matches_coherence = random.sample(signal_matches_coherence, N_comb_to_use)
      N_tuples_to_compute = len(signal_matches_coherence)

    list_avg_error, list_global_error = [], []
    for i_tuple in range(N_tuples_to_compute):

      matching_tuple = signal_matches_coherence[i_tuple]
      i_current = matching_tuple[0]
      j_current = matching_tuple[1]

      

      if i_tuple > 0:
         previous_mag_coh = current_mag_coh.copy()


      f, pxy = signal.csd(signals_to_augment[i_current],
                            signals_to_augment[j_current],
                            nperseg=len(signals_to_augment[i_current]),
                            return_onesided=True)
      
      current_mag_coh = np.abs(pxy)
      mag_coherence_list.append(current_mag_coh)
      #phase_coherence_list.append(np.angle(pxy) / np.pi)
      freq_list.append(f)

      # To avoid too long computations, implement a halt mechanism
      # that stops coherence mag computation if changes in the avg
      # mag coh cuver for every point becomes smaller than a given eps
      # (we will be computing the avg mag curve anyway out of the
      # collection of coherence curves stored in mag_coherence_list)

      

      if i_tuple > MIN_NUM_TUPLES:
         
         error_coh_approx = np.nanmean(np.abs(current_mag_coh - previous_mag_coh))

         # avg mean abs error for current combination of signals creating coh curve
         list_avg_error.append(error_coh_approx) 

         # Mean avg. mean abs for all combinations
         list_global_error.append(np.nanmean(list_avg_error)) 

         if len(list_global_error) > N_POINTS_MAG_COH_CONV_HAT:
          error_criterion = np.nanmean(np.diff(list_global_error[-N_POINTS_MAG_COH_CONV_HAT:]))

          if i_tuple % 1000 == 0:
            print("i_tuple is " + str(i_tuple) + "of a max of " + str(MAX_COH_CURVES) + "steps (but it can halt before that, if avg error converge), avg error is " + str(error_criterion))
          
          if error_criterion < EPS_TO_HALT_MAG_COH_COMP or i_tuple > MAX_COH_CURVES:
            break
         

      

      

    freq_points = np.mean(freq_list, axis=0).tolist()

    return mag_coherence_list, phase_coherence_list, freq_points, signal_matches_coherence, list_avg_error


def compute_mod_surro(input_signal, fs, sorted_freq, frac_freq_shuffle, n_replicates):
    """
    Computes the modified surrogate for a given input signal 
    considering the array of relevant frequencies for classes
    computed earlier. 
    
    Args:
    input_signal: Signal (numpy array) to compute the mod surro.
    
    fs: The sampling frequency to consider.
    
    sorted_freq: array of freq values sorted from the most relevant
    to the least relevant frequencies of the class.
    
    frac_freq_shuffle: fraction of points of sorted_freq to shuffle.
    The higher the fraction the more freqs. will be shuffled.
    In the limit, if the fraction equals one, all frequencies are
    shuffled and the mod surro becomes the original surrogate.
    
    n_replicates: number of modified surrogates to create for
    each signal.
    
    
    Returns:
    array_surro: An array with n_replicates elements where each
    element is one modified surrogate computed from the input_signal.
    
    """

    
    current_fft = np.fft.fft(input_signal)
    current_freq_points = np.fft.fftfreq(len(input_signal), d=1./fs)
    current_freq_points = current_freq_points[current_freq_points >= 0]
    current_freq_idx = np.where(current_freq_points >= 0)
    current_mag = np.abs(current_fft[current_freq_idx])
    current_mag = np.concatenate((current_mag,np.zeros(len(current_mag))), axis = 0)
    current_phase = np.angle(current_fft)
    
    freq_values_to_consider = sorted_freq[:int(frac_freq_shuffle*len(sorted_freq))]
    
    
    list_surrogates = []
    for i_rep in range(n_replicates):

        for freq_value in freq_values_to_consider:

            # Find the closest freq value
            closest_freq_value, id_argmin = find_closest_freq_value(current_freq_points, freq_value)

            current_phase[id_argmin] = np.random.uniform(0, np.pi)

        # current_phase = np.random.rand(len(current_phase))*2*np.pi
        modified_surro = np.real(np.fft.ifft(2*current_mag*np.exp((1j)*current_phase)))

        list_surrogates.append(modified_surro)
        
    array_surro = np.array(list_surrogates)
        
    return array_surro









