import csv
import random
from pathlib import Path

# --- Experiment Parameters ---
# IMPORTANT: These must match the filenames in your /stimuli/ folder
LURE_STIMULI = ['stimuli/face_lure.png', 'stimuli/scissor_lure.png', 'stimuli/zebra_lure.png', 'stimuli/banana_lure.png']
NEW_STIMULI = ['stimuli/new1.png', 'stimuli/new2.png', 'stimuli/new3.png', 'stimuli/new4.png']
NOT_IN_SEQUENCE_STIMULI = LURE_STIMULI + NEW_STIMULI

NUM_PARTICIPANTS = 99 # How many learning files do you have?
TOTAL_TRIALS = 96
NUM_BLOCKS = 3
TRIALS_PER_BLOCK = TOTAL_TRIALS // NUM_BLOCKS

# --- Setup Paths ---
learning_conditions_path = Path('conditions_learning')
output_conditions_path = Path('conditions')
output_conditions_path.mkdir(exist_ok=True)

# --- Main Generation Loop ---
for p_id in range(1, NUM_PARTICIPANTS + 1):
    print(f"Generating mental simulation files for participant {p_id}...")

    # 1. Read the learning CSV to reconstruct the sequence A -> B -> C -> D
    learning_file = learning_conditions_path / f'learning_conditions_{p_id}.csv'
    try:
        with open(learning_file, 'r', newline='') as f:
            # Read all learning pairs into a list of dictionaries
            learning_pairs = list(csv.DictReader(f))
            
            # Find the specific rows for each pair
            row_AB = next(row for row in learning_pairs if row['pair_name'] == 'A_B')
            row_BC = next(row for row in learning_pairs if row['pair_name'] == 'B_C')
            row_CD = next(row for row in learning_pairs if row['pair_name'] == 'C_D')

            # Reconstruct the sequence from the pairs
            stim_A = row_AB['stim1_img']
            stim_B = row_AB['stim2_img']
            stim_C = row_BC['stim2_img']
            stim_D = row_CD['stim2_img']
            
            # Create the final ordered list of in-sequence stimuli
            in_sequence_stimuli = [stim_A, stim_B, stim_C, stim_D]
            print(f"  > Reconstructed sequence: {in_sequence_stimuli}")

            # Sanity check: ensure the chain is correct
            if not (row_AB['stim2_img'] == row_BC['stim1_img'] and row_BC['stim2_img'] == row_CD['stim1_img']):
                print(f"  ! WARNING: Sequence chaining mismatch in {learning_file}. Check your file.")

    except FileNotFoundError:
        print(f"  ! ERROR: Learning file not found: {learning_file}. Skipping this participant.")
        continue
    except StopIteration:
        print(f"  ! ERROR: Could not find all pairs (A_B, B_C, C_D) in {learning_file}. Skipping.")
        continue


    # 2. Create a balanced list of all trial types (This part is the same as before)
    # We need 96 trials: 48 forward, 48 backward; 48 in-sequence, 48 not-in-sequence
    cue_directions = ['forward'] * (TOTAL_TRIALS // 2) + ['backward'] * (TOTAL_TRIALS // 2)
    probe_types = ['in-sequence'] * (TOTAL_TRIALS // 2) + ['not-in-sequence'] * (TOTAL_TRIALS // 2)

    # Combine and shuffle to create a randomized trial list
    trial_specs = list(zip(cue_directions, probe_types))
    random.shuffle(trial_specs)

    # 3. Flesh out the trial details using the reconstructed sequence
    final_trials = []
    for i, (cue_dir, probe_type) in enumerate(trial_specs):
        block_num = (i // TRIALS_PER_BLOCK) + 1
        
        # Set cue text based on direction
        cue_text = '1 ->' if cue_dir == 'forward' else '<- 4'


        if p_id % 2 == 0:
            # Set probe image and correct response
            if probe_type == 'in-sequence':
                probe_image = random.choice(in_sequence_stimuli)
                correct_response = '1'
            else: # 'not-in-sequence'
                probe_image = random.choice(NOT_IN_SEQUENCE_STIMULI)
                correct_response = '2'

            final_trials.append({
                'block': block_num,
                'cue_direction': cue_dir,
                'cue_text': cue_text,
                'probe_image': probe_image,
                'correct_response': correct_response
            })
        else : 
            # Set probe image and correct response
            if probe_type == 'in-sequence':
                probe_image = random.choice(in_sequence_stimuli)
                correct_response = '2'
            else: # 'not-in-sequence'
                probe_image = random.choice(NOT_IN_SEQUENCE_STIMULI)
                correct_response = '1'

            final_trials.append({
                'block': block_num,
                'cue_direction': cue_dir,
                'cue_text': cue_text,
                'probe_image': probe_image,
                'correct_response': correct_response
            })

    # Write all trials to a single CSV in the main /conditions folder
    output_filename = output_conditions_path / f'p{p_id}_mental_sim_conditions.csv'
    with open(output_filename, 'w', newline='') as f:
        fieldnames = ['block', 'cue_direction', 'cue_text', 'probe_image', 'correct_response']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_trials)

print("\nMental simulation condition files generated successfully in the /conditions/ folder.")