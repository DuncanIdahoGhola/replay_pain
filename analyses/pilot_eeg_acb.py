#start with packages that will be usefull

import pandas as pd
import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from numpy.fft import fft, fftfreq
from scipy import signal
from mne.fixes import minimum_phase
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs


#define directories 
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
eeg_data_file = project_root / 'eeg_files'

sub_directory = list(eeg_data_file.glob('sub-*'))

for sub in sub_directory:
    func_loc_eeg = sub / 'func_loc' 
    func_loc_eeg_file = list(func_loc_eeg.glob('cmapsfuncloc*.vhdr'))
    assert len(func_loc_eeg_file) == 1
    rest_state_eeg = sub / 'rest_state'
    rest_sate_eeg_file = list(rest_state_eeg.glob('cmaps_rest_state*.vhdr'))
    assert len(rest_sate_eeg_file) == 1 
    learn_prob_eeg = sub / 'learn_prob'
    learn_prob_eeg_file = list(learn_prob_eeg.glob('cmaps_learn_prob*.vhdr'))
    assert len (learn_prob_eeg_file) == 1
    mental_stim_eeg = sub / 'mental_stim'
    mental_stim_eeg_file = list (mental_stim_eeg.glob('cmaps_cued_stim*.vhdr'))
    assert len (mental_stim_eeg_file) == 1 
    rest_sate_eeg_file = rest_sate_eeg_file [0]
    learn_prob_eeg_file = learn_prob_eeg_file [0]
    func_loc_eeg_file = func_loc_eeg_file [0]
    mental_stim_eeg_file = mental_stim_eeg_file [0]


    #We start with rest_state as pratice
    raw = mne.io.read_raw_brainvision(rest_sate_eeg_file, eog=['HEOG', 'VEOG'], misc=['ECG'] , preload=True )
    # set the helmet thingy
    raw.set_montage('standard_1020')
    
    #resample too 500  
    

    #set the filer 
    #raw.filter(1, 40)
    #rename ECG channel and find epochs from raw ECG
    raw.set_channel_types({'ECG': 'ecg'})
    ecg_evoked = create_ecg_epochs(raw).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    ecg_evoked.plot_joint()

    #Find eyes artifacts from raw EOG
    eog_evoked = create_eog_epochs(raw).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    eog_evoked.plot_joint()

    #we might need to add resampling later around here 
   
   #add filter copy of raw for ICA 
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

    #try ICA with n components
    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_raw)
    ica

    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices

    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)

    # plot diagnostics
    ica.plot_properties(raw, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw, show_scrollbars=False)

    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    ica.plot_sources(eog_evoked)

    #same for ecg 
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')
    ica.exclude.extend(ecg_indices)  # Combine EOG and ECG

    #Apply ICA on continous data
    ica.apply(raw)
    raw.filter(1,40)
    raw.plot()
    raw.plot_psd(fmax=50)








