#Create conditions files for learning phase
#we have 4 stimuli : A, B, C, D
#We want to randomized which stimuli is A, B, C, D
#We then want to present A-B, B-C, C-D
import pandas as pd
import random
import os

# name the 4 stimuli

zebra = 'stimuli/zebra.png'
face = 'stimuli/face.png'
banana = 'stimuli/banana.png'
scissor = 'stimuli/scissor.png'




#Randomize which stimuli is A, B, C, D
stimuli = [zebra, face, banana, scissor]

for i in range(1, 100):
    # Randomly shuffle the stimuli
    random.seed(i)
    random.shuffle(stimuli)

    # Create a DataFrame with the randomized stimuli
    df = pd.DataFrame({
        'A': [stimuli[0]],
        'B': [stimuli[1]],
        'C': [stimuli[2]],
        'D': [stimuli[3]]
    })

    #Change data frame so that we have 3 rows in csv, pair_name, stim1_img and stim2_img
    df = pd.DataFrame({
        'pair_name': ['A_B', 'B_C', 'C_D'],
        'stim1_img': [stimuli[0], stimuli[1], stimuli[2]],
        'stim2_img': [stimuli[1], stimuli[2], stimuli[3]]
    })
    # Save the DataFrame to a CSV file
    output_dir = 'conditions_learning'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'learning_conditions_{i}.csv'), index=False)