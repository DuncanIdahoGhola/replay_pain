import pandas as pd
import random
import os

# --- Configuration ---
# Define the input and output directories here.
# This makes it easy to change them later.
LEARNING_CONDITIONS_DIR = 'conditions_learning'
PROBE_CONDITIONS_DIR = 'conditions_probe'

def generate_mock_learning_files(num_files=100):
    """
    Generates mock learning condition files into the specified directory.
    This is for demonstration if you don't have the files yet.
    """
    print(f"--- Generating {num_files} mock learning files for demonstration ---")
    
    # Create the learning directory if it doesn't exist
    os.makedirs(LEARNING_CONDITIONS_DIR, exist_ok=True)

    base_stimuli = [
        'stimuli/face.png', 'stimuli/ciseau.png',
        'stimuli/zèbre.png', 'stimuli/banane.png'
    ]

    for i in range(1, num_files + 1):
        random.shuffle(base_stimuli)
        pairs = [
            {'pair_name': 'A_B', 'stim1_img': base_stimuli[0], 'stim2_img': base_stimuli[1]},
            {'pair_name': 'B_C', 'stim1_img': base_stimuli[1], 'stim2_img': base_stimuli[2]},
            {'pair_name': 'C_D', 'stim1_img': base_stimuli[2], 'stim2_img': base_stimuli[3]}
        ]
        df = pd.DataFrame(pairs)
        
        # *** CHANGE: Use os.path.join to save into the correct folder ***
        filename = os.path.join(LEARNING_CONDITIONS_DIR, f'learning_conditions_{i}.csv')
        df.to_csv(filename, index=False)
        
    print(f"--- Mock files generated in '{LEARNING_CONDITIONS_DIR}/' folder. ---\n")


def create_probe_file_for_participant(participant_id):
    """
    Reads a learning file from the input directory and creates a 
    corresponding probe file in the output directory.
    """
    # *** CHANGE: Build the full path for the input file ***
    learning_file = os.path.join(LEARNING_CONDITIONS_DIR, f'learning_conditions_{participant_id}.csv')
    
    # *** CHANGE: Build the full path for the output file ***
    output_file = os.path.join(PROBE_CONDITIONS_DIR, f'probe_conditions_{participant_id}.csv')

    if not os.path.exists(learning_file):
        print(f"Warning: {learning_file} not found. Skipping participant {participant_id}.")
        return

    # 1. Read the learning file and determine the sequence
    df_learn = pd.read_csv(learning_file)
    sequence = [df_learn['stim1_img'][0]] + df_learn['stim2_img'].tolist()
    all_stimuli_set = set(sequence)
    possible_targets = sequence[:-1]

    trials = []
    cue_text = "Qu'est-ce qui vient ensuite ?"

    # 2. Generate 6 MATCH trials
    for _ in range(2):
        for i, target_img in enumerate(possible_targets):
            correct_probe_img = sequence[i + 1]
            trials.append({
                'target_img': target_img, 'probe_img': correct_probe_img,
                'cue_text': cue_text, 'is_match': 'match'
            })

    # 3. Generate 6 NON-MATCH trials
    for i, target_img in enumerate(possible_targets):
        correct_probe_img = sequence[i + 1]
        possible_non_matches = list(all_stimuli_set - {target_img, correct_probe_img})
        trials.append({
            'target_img': target_img, 'probe_img': possible_non_matches[0],
            'cue_text': cue_text, 'is_match': 'non-match'
        })
        trials.append({
            'target_img': target_img, 'probe_img': possible_non_matches[1],
            'cue_text': cue_text, 'is_match': 'non-match'
        })
        
    # 4. Randomize the order of all 12 trials
    random.shuffle(trials)

    # 5. Create a DataFrame and save to the specified output CSV
    df_probe = pd.DataFrame(trials)
    df_probe.to_csv(output_file, index=False)
    
    print(f"Successfully created {output_file} from {learning_file}.")


# --- Main Execution ---
if __name__ == "__main__":
    # If you need to generate test files, uncomment the next line.
    # generate_mock_learning_files(num_files=100)

    # *** CHANGE: Create the output directory before starting the loop. ***
    # The exist_ok=True argument prevents an error if the folder already exists.
    print(f"Ensuring output directory '{PROBE_CONDITIONS_DIR}/' exists...")
    os.makedirs(PROBE_CONDITIONS_DIR, exist_ok=True)
    print("...directory ready.\n")

    # Now, loop through all participants and generate their probe files
    for i in range(1, 101):
        create_probe_file_for_participant(i)

    print("\n--- All probe condition files have been generated. ---")
    
    # You can inspect one of the generated files to see the output
    example_file_path = os.path.join(PROBE_CONDITIONS_DIR, 'probe_conditions_1.csv')
    print(f"\nExample output from '{example_file_path}':")
    if os.path.exists(example_file_path):
        print(pd.read_csv(example_file_path))