import pandas as pd
import random
import os

#we want to find for each learning conditions what order the stimuli is in 
#Find path to the conditions_learning folders

cond_learning_path = os.path.join(os.getcwd(), 'conditions_learning')

# Get a list of all files in the conditions_learning directory
files = os.listdir(cond_learning_path)
# Filter the list to include only .csv files
csv_files = [f for f in files if f.endswith('.csv')]


#to make it easy get only the first file
for files in csv_files:
    # Read the CSV file into a DataFrame
    random.seed(files)
    df = pd.read_csv(os.path.join(cond_learning_path, files))
    
    #make a list of the order of the stimuli (A_B_C_D)
    stim_1 = df['stim1_img'][0]
    stim_2 = df['stim1_img'][1]
    stim_3 = df['stim1_img'][2]
    stim_4 = df['stim2_img'][2]

    #create a dataframe with 12 rows of target_img = stim_1, stim_2, stim_3, stim_4 3 of each randoized order
    target_img = [stim_1, stim_2, stim_3, stim_4] * 3
    
    match_image_1 = stim_2
    match_image_2 = stim_3
    match_image_3 = stim_4
    match_image_4 = stim_1

    match_img = [match_image_1, match_image_2, match_image_3, match_image_4] * 3

    first_six = match_img[:6]
    rest = match_img[6:]
    def deranged_shuffle(lst):
        while True:
            shuffled = lst[:]
            random.shuffle(shuffled)
            # Check that no element is in its original position
            if all(shuffled[i] != lst[i] for i in range(len(lst))):
                return shuffled

    first_six_deranged = deranged_shuffle(first_six)
    match_img = first_six_deranged + rest
    #create data frame with both lists
    df_target = pd.DataFrame({'target_img': target_img, 'prob_img': match_img})

    #Add cue_text column with the same value for all rows
    df_target['cue_text'] = '->...->...->?'

    #Create column prob_img with value = the image next in line to target_img for 6 rows and 'blank' for the other 6 rows
    #Replace df_target blanks with random target_img which are not in target_img and are not next in line to target_img
    # Create a list of images that are not the target image
    #shuffle the rows of df_target
    df_target = df_target.sample(frac=1).reset_index(drop=True)

    for i in range(len(df_target)):
        if df_target.loc[i, 'target_img'] == stim_1 and df_target.loc[i, 'prob_img'] == stim_2:
            df_target.loc[i, 'is_match'] = 'match'
        elif df_target.loc[i, 'target_img'] == stim_2 and df_target.loc[i, 'prob_img'] == stim_3:
            df_target.loc[i, 'is_match'] = 'match'
        elif df_target.loc[i, 'target_img'] == stim_3 and df_target.loc[i, 'prob_img'] == stim_4:
            df_target.loc[i, 'is_match'] = 'match'
        elif df_target.loc[i, 'target_img'] == stim_4 and df_target.loc[i, 'prob_img'] == stim_1:
            df_target.loc[i, 'is_match'] = 'match'
        else:
            df_target.loc[i, 'is_match'] = 'no_match'
    # Save the DataFrame to a new CSV file
    output_dir = 'conditions_probe'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'probe_{files}')
    df_target.to_csv(output_file, index=False)
