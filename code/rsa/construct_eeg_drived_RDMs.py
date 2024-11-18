################################ EEG RDMS ################################ 
#
# This script constructs Representational Dissimilarity Matrices (RDMs) derived from EEG data.
#
# Author: Denise Kittelmann 

################################ LOAD PACKAGES ################################ 

import mne
import glob
import os
import scipy.io
from scipy.spatial.distance import cosine
import numpy as np
from pymatreader import read_mat
import matplotlib.pyplot as plt


################################ HELPER FUNCTIONS ################################ 


# create condlist
conditionlist = {
    'leading_Barn': 1,
    'leading_beach': 2,
    'leading_library': 3,
    'leading_restaurant': 4, 
    'leading_cave': 5,
    'trailing_church': 6,
    'trailing_conference_room': 7,
    'trailing_castle': 8, 
    'trailing_forest': 9      
}


# create category dict
#Idea: we create a customised event_id dict that allows us to later just extract the trailing images corresponding to our transitional probabiliy categories

# Mapping VALID CONDITIONS (75%): 
    # leading_barn -> trailing_church
    # leading_beach -> trailing_church
    # leading_library -> trailing_conference_room
    # leading_restaurant ->  trailing_conference_room

# Mapping INVALID CONDITIONS (25%): 
    # leading_barn -> trailing_conference_room
    # leading_beach -> trailing_conference_room
    # leading_library -> trailing_church
    # leading_restaurant -> trailing_church
    
# Mapping CONTROL CONDITIONS (50%): 
    # leading_cave -> trailing_castle
    # leading_cave -> trailing_forest
    
category_dict = {
    0: (1, 6), # valid conditions 75 %
    1: (2, 6), # leading_beach -> trailing_church
    2: (3, 7), # leading_library -> trailing_conference_room
    3: (4, 7),  # leading_restaurant ->  trailing_conference_room
    4: (1, 7), # invalid conditions 0.25 %
    5: (2, 7),
    6: (3, 6), 
    7: (4, 6), 
    8: (5, 8), # control conditions 50 %
    9: (5, 9)
}


rdm_dict = {
    0: (0,1), # 0.75%
    1: (2,3), # 0. 75 % 
    2: (4,5), # 0.25 % 
    3: (6,7), # 0.25 %
    4: (8,), # 0.5 % 
    5: (9,)  # 0.5 %  
}




def convert_trialinfo(data, mapping):
    '''This function converts condition label strings into integer based on the mapping in conditionlist.
        Input: data, condition mapping
        Returns: List of int
    '''    
    if "trialinfo" in data:
        trialinfo_labels = data["trialinfo"]
    else:
        raise KeyError("'trialinfo' field is missing in data['fD']")

    return np.array([mapping[cond] for cond in trialinfo_labels])


def map_events(event_id, category_dict):
    """
    This function creates a customised event_id dict that allows us to later just extract the trailing images 
    corresponding to our transitional probabiliy categories
    Maps events based on specified (leading, trailing) pairs in category_dict
    
    Parameters:
    - event_id 
    - category_dict (dict)
    
    Returns:
    - list: Mapped event categories based on leading-trailing pairs
    """
    # Initialize event_maps with -1 for each event
    event_maps = [-1] * len(event_id)
    
    # Iterate over event IDs, starting from the second event
    for idx in range(1, len(event_id)):
        # Check if the current and previous event form a valid (leading, trailing) pair
        for key, (leading, trailing) in category_dict.items():
            if idx > 0 and (event_id[idx - 1] == leading) and (event_id[idx] == trailing):
                event_maps[idx] = key
    
    return event_maps




def get_unique_event_ids(event_id):
    """
    Retrieves unique event trial ids, excluding the trial id -1 (leading images)
    
    Parameters:
    - event_id (list or array-like)
    
    Returns:
    - np arrays
    """
    # Get unique event IDs and filter out -1
    unique_event_ids = np.unique(event_id)
    unique_event_ids = unique_event_ids[unique_event_ids != -1]
    
    return unique_event_ids


################################ PATHS ################################ 


results_path_eeg = "/Users/denisekittelmann/Documents/Python/BiMoL/results/rdms/eeg_RDM/"


################################ COMPUTE EEG RDM ################################ 

################################ COMPUTE EEG RDM ################################ 

# Define dir where EEG can be found 
dir = "/Users/denisekittelmann/Documents/Python/BiMoL/data/EEG/"

# Create iterator
ids = [pid for pid in range(1, 32) if pid != 20] 

# Loop over all participants and compute EEG RDMs for each hypothesis
for pid in ids:
    
    print(pid)

    fname = os.path.join(dir, f"eTadff_sub{pid:02d}.mat")
    print(f"Loading data for participant {pid} from file {fname}")
    data = read_mat(fname)["fD"]

    data["trialinfo"] = convert_trialinfo(data, conditionlist)
    print(data["trialinfo"])
    

    # read the fields that we will need later
    epochs_data = np.array(data["trial"]) * 1e-6 # muV
    tmin = data['time'][0][0]
    sfreq = int(1. / (data['time'][0][1] - tmin))
    ch_names = data["label"]
    ch_types = data["elec"]["chantype"]
    montage = mne.channels.make_standard_montage("biosemi64", head_size=0.095)  
    # event_dict = {}

    event_id = list(data["trialinfo"])
    events = np.stack([np.arange(len(event_id)),
                    np.zeros_like(event_id), 
                    event_id], axis=1)
    
    # create the info field
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create an MNE Epochs object
    epochs = mne.EpochsArray(data=epochs_data, info=info, tmin=tmin,
                            events=events, event_id=event_id) 

    # set the montage for correct channel positions
    epochs = epochs.set_montage(montage)

    
    event_id = map_events(event_id, category_dict)
    print("event_id:",event_id)


    ################################ COMPUTE EEG RDM ################################ 
    print(f"################################ Preparing to compute RDMs for participant {pid:02d} ################################")
    
    # Construct empty RDMs
    n_categories = len(rdm_dict)
    rdm_b1 = np.zeros((n_categories, n_categories))
    rdm_b2 = np.zeros((n_categories, n_categories))
    rdm_early = np.zeros((n_categories, n_categories))
    rdm_late = np.zeros((n_categories, n_categories))
    
    # Define time windows for H2
    early_start, early_end = 0.128, 0.180
    late_start, late_end = 0.280, 0.296

    epochs_early = epochs.copy().crop(tmin=early_start, tmax=early_end)
    epochs_late = epochs.copy().crop(tmin=late_start, tmax=late_end)
    
    b1 = epochs[:len(epochs) // 2]
    b2 = epochs[len(epochs) // 2:]


    # Compute EEG RDMs for each participant
    print(f"################################ Starting to compute RDMs for participant {pid:02d} ################################")
    
    for i, (category_i, idx_i) in enumerate(rdm_dict.items()):
        for j, (category_j, idx_j) in enumerate(rdm_dict.items()):
            
            if isinstance(idx_i, int): # to account for cat 4 & 5 which are just single integers
                idx_i = [idx_i]
                
            if isinstance(idx_j, int):
                idx_j = [idx_j]
            
            epochs_i = [b1[event_name] for event_name in idx_i]
            mean_representation_i = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_i], axis=0)
            
            epochs_j = [b1[event_name] for event_name in idx_j]
            mean_representation_j = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_j], axis=0)
            
            rdm_b1[i, j] = cosine(mean_representation_i, mean_representation_j)
            

    print(f"################################ Finished computing RDM_B1 for participant {pid:02d}. STARTING DO COMPUTE RDM_B2. ################################")
    
    for i, (category_i, idx_i) in enumerate(rdm_dict.items()):
        for j, (category_j, idx_j) in enumerate(rdm_dict.items()):
            
            if isinstance(idx_i, int): # to account for cat 4 & 5 which are just single integers
                idx_i = [idx_i]
                
            if isinstance(idx_j, int):
                idx_j = [idx_j]
            
            epochs_i = [b2[event_name] for event_name in idx_i]
            mean_representation_i = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_i], axis=0)
            
            epochs_j = [b2[event_name] for event_name in idx_j]
            mean_representation_j = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_j], axis=0)
            
            rdm_b2[i, j] = cosine(mean_representation_i, mean_representation_j)
            np.save(os.path.join(results_path_eeg, f"eeg_rdm_b2_sub{pid:02d}.npy"), rdm_b2)
 
    print(f"################################ Finished computing RDM_B2 for participant {pid:02d}. STARTING DO COMPUTE RDM_EARLY. ################################")

    for i, (category_i, idx_i) in enumerate(rdm_dict.items()):
        for j, (category_j, idx_j) in enumerate(rdm_dict.items()):
            
            if isinstance(idx_i, int): # to account for cat 4 & 5 which are just single integers
                idx_i = [idx_i]
                
            if isinstance(idx_j, int):
                idx_j = [idx_j]
            
            epochs_i = [epochs_early[event_name] for event_name in idx_i]
            mean_representation_i = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_i], axis=0)
            
            epochs_j = [epochs_early[event_name] for event_name in idx_j]
            mean_representation_j = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_j], axis=0)
            
            rdm_early[i, j] = cosine(mean_representation_i, mean_representation_j)
            np.save(os.path.join(results_path_eeg, f"eeg_rdm_late_sub{pid:02d}.npy"), rdm_late)

    print(f"################################ Finished computing RDM_EARLY for participant {pid:02d}. STARTING DO COMPUTE RDM_LATE. ################################")
    
    for i, (category_i, idx_i) in enumerate(rdm_dict.items()):
        for j, (category_j, idx_j) in enumerate(rdm_dict.items()):
            
            if isinstance(idx_i, int): # to account for cat 4 & 5 which are just single integers
                idx_i = [idx_i]
                
            if isinstance(idx_j, int):
                idx_j = [idx_j]
            
            epochs_i = [epochs_late[event_name] for event_name in idx_i]
            mean_representation_i = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_i], axis=0)
            
            epochs_j = [epochs_late[event_name] for event_name in idx_j]
            mean_representation_j = np.mean([epoch.average().get_data().reshape(-1) for epoch in epochs_j], axis=0)
            
            rdm_late[i, j] = cosine(mean_representation_i, mean_representation_j)
            np.save(os.path.join(results_path_eeg, f"eeg_rdm_early_sub{pid:02d}.npy"), rdm_early)
            
    print(f"################################ Finished computing RDM_B2 for participant {pid:02d} ################################")

    print(f"################################ Saving RDMs for participant {pid:02d} ################################")
    np.save(os.path.join(results_path_eeg, f"eeg_rdm_b1_sub{pid:02d}.npy"), rdm_b1)
    np.save(os.path.join(results_path_eeg, f"eeg_rdm_b2_sub{pid:02d}.npy"), rdm_b2)
    np.save(os.path.join(results_path_eeg, f"eeg_rdm_early_sub{pid:02d}.npy"), rdm_early)
    np.save(os.path.join(results_path_eeg, f"eeg_rdm_late_sub{pid:02d}.npy"), rdm_late)


################################ PLOT EEG RDMs ################################ 


rdm = np.load('/Users/denisekittelmann/Documents/Python/BiMoL/results/rdms/eeg_RDM/eeg_rdm_late_sub01.npy')
plt.imshow(rdm, cmap='viridis') # twilight_shifted inferno plasma 
#plt.xticks([])
#plt.yticks([])
plt.colorbar()
plt.show()
plt.close()
    
        

