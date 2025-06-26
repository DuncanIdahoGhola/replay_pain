#start with packages that will be usefull

import pandas as pd
import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#define directories 
#find main data 

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
data_file = project_root / 'replay_pilot'/'data' 


#create deriv_folder
deriv_path = Path(data_file) / "derivatives"
deriv_path.mkdir(parents=True, exist_ok=True)



#find sub_dir

sub_directory = list(data_file.glob('sub-*'))

#remove unwanted files here

unwanted = ['sub-088']

all_participant_dfs = []

#create loop
for sub in sub_directory : 
    if sub.name in unwanted :
        continue
    #find all files in path
    rest_state_dir = sub / 'rest_state' 
    learn_prob_dir = sub / 'learn_prob'
    func_loc_dir = sub / 'func_loc'
    cued_stim_dir = sub / 'cued_mental_stim'

    #find all csv file 
    rest_state_file = list(rest_state_dir.glob('sub-*rest_state*.csv'))
    learn_prob_file = list(learn_prob_dir.glob('sub-*learn_prob*.csv'))
    func_loc_file = list(func_loc_dir.glob('sub-*FunctionalLocalizer*.csv'))
    cued_stim_file = list(cued_stim_dir.glob('sub-*cued_mental_stim*.csv'))

    #ensure only 1 csv file for each
    assert len(rest_state_file) == 1
    assert len(learn_prob_file) == 1
    assert len(func_loc_file) == 1
    assert len(cued_stim_file) == 1
    #undo list
    func_loc_file = func_loc_file[0]
    cued_stim_file = cued_stim_file[0]
    learn_prob_file = learn_prob_file[0]
    rest_state_file = rest_state_file[0]


    #Start analyses on rest state (we do not need anything here)

    #continue with func_loc
    func_loc_df = pd.read_csv(str(func_loc_file))
    #drop first two rwos of df and last row and then find the correct answesr %
    func_loc_df = func_loc_df.drop(func_loc_df.index[[0,1,-1]])

    correct_answers_percentage_func = func_loc_df['response_correct'].fillna(value=0).sum() / len(func_loc_df) * 100

    #go to learn prob performance

    learn_prob_df = pd.read_csv(str(learn_prob_file))
    good_answer_percentage_learn = learn_prob_df['good_answer_counter'].dropna().iloc[-1] / len(learn_prob_df['good_answer_counter'].dropna()) * 100

    #go to cued_stim performance

    cued_stim_df = pd.read_csv(str(cued_stim_file))

    stim_performance_percentage = cued_stim_df['trials_loop.key_resp_probe.corr'].dropna().sum() / len(cued_stim_df['trials_loop.key_resp_probe.corr'].dropna()) * 100
    
    #we can add mean ratings for vividness of mental stim if we want here, will do later if we analyse it

    #add all variables to a new data frame

    new_df = pd.DataFrame({ 'participant' : [sub.name],
                            'func_loc_performance' : [correct_answers_percentage_func],
                           'learn_prob_performance' : [good_answer_percentage_learn],
                           'cued_stim_performance' : [stim_performance_percentage],
                          })
    
    all_participant_dfs.append(new_df)


final_df = pd.concat(all_participant_dfs, ignore_index=True)

final_df.to_csv(deriv_path / 'behaviour_results.csv',index=False)