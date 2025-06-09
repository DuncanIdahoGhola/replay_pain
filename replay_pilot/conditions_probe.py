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
    random.shuffle(target_img)
    df_target = pd.DataFrame({'target_img': target_img})

    #Add cue_text column with the same value for all rows
    df_target['cue_text'] = '->...->...->?'

    #Create column prob_img with same value as target_img for 6 rows and 'blank' for the other 6 rows
    prob_img = target_img[:6] + ['blank'] * 6
    df_target['prob_img'] = prob_img

    #Replace df_target blanks with random target_img which are not in target_img
    for i in range(len(df_target)):
        if df_target['prob_img'][i] == 'blank':
            df_target['prob_img'][i] = random.choice([img for img in target_img if img != df_target['target_img'][i]])

    #shuffle the rows of df_target
    df_target = df_target.sample(frac=1).reset_index(drop=True)     

    #Add a corrAns column, value = 1 if prob_img is the same as target_img, else 2
    df_target['corrAns'] = df_target.apply(lambda row: 1 if row['prob_img'] == row['target_img'] else 2, axis=1)   

    # Save the DataFrame to a new CSV file
    output_dir = 'conditions_probe'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'probe_{files}')
    df_target.to_csv(output_file, index=False)
