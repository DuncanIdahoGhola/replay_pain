#this code will create a singular train condition csv file used for the training run of func_loc

import pandas as pd
import os
#create a dataframe

df = pd.DataFrame()

images = ['training_1', 'training_2'] * 2

image_paths = [f'/stimuli/{i}.png' for i in images] 

#create 4 rows in data frame that have the images
#add stimuli/images.png
df['images'] = image_paths 

#do the same with words
#create match trials and non match 


match_words = ['amande', 'valise']
non_match_words = ['valise','amande']

words = match_words + non_match_words

df['words'] = words

#add a match column 

match = ['match'] * 2
n_match = ['n_match'] * 2 
df['match'] = match + n_match

#multiply the rows by a number we hope participants do not reach (lets say 5)

df = pd.concat([df] * 5, ignore_index=True)


#shuffle the data frame

df = df.sample(frac=1).reset_index(drop=True)

#save to csv in new file path 

OUTPUT_DIR = "train_cond"
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_path = os.path.join(OUTPUT_DIR, f"train_cond.csv")
df.to_csv(file_path, index=False)