import pandas as pd
import random


#code generating the localizer conditions

#set seed 
random.seed(42)
images = ['face', 'zebra', 'banana', 'scissor']
rows = []

for img in images:
    # MATCH trials
    for _ in range(36):
        rows.append({
            'image_file': f'stimuli/{img}.png',
            'presented_word': img,
            'is_match': 'match',
            'corrAns': 1,
            'blank_duration': round(random.uniform(1.0, 2.0), 2),
            'iti_duration': round(random.uniform(1.0, 3.0), 2)
        })

    # MISMATCH trials
    mismatches = [w for w in images if w != img]
    for _ in range(36):
        word = random.choice(mismatches)
        rows.append({
            'image_file': f'stimuli/{img}.png',
            'presented_word': word,
            'is_match': 'mismatch',
            'corrAns': 2,
            'blank_duration': round(random.uniform(1.0, 2.0), 2),
            'iti_duration': round(random.uniform(1.0, 3.0), 2)
        })

# Shuffle all trials making sure that no more than 2 trials with the same images are adjacent

def shuffle_trials_corrected_outer_loop(trials, max_attempts=100000):
    for attempt in range(max_attempts):
        random.shuffle(trials)
        violation_found = False
        for i in range(len(trials) - 2):
            if (trials[i]['image_file'] == trials[i + 1]['image_file'] == trials[i + 2]['image_file']):
                violation_found = True
                break  # Found a violation, break inner loop to reshuffle
        
        if not violation_found:
            print(f"Constraint satisfied after {attempt + 1} attempts.")
            return trials  # No violation found in this shuffle, so it's good
    
    print(f"Warning: Could not satisfy shuffle constraint after {max_attempts} attempts. Returning the last attempt.")
    return trials # Return the list even if constraint not met after max_attempts

rows = shuffle_trials_corrected_outer_loop(rows)

# Save to Excel
df = pd.DataFrame(rows)
df.to_excel("localizer_conditions.xlsx", index=False)


# check the generated DataFrame - how many match and mismatch trials?
match_count = df[df['is_match'] == 'match'].shape[0]
mismatch_count = df[df['is_match'] == 'mismatch'].shape[0]
print(f"Match trials: {match_count}")
print(f"Mismatch trials: {mismatch_count}")

