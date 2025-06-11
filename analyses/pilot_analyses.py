import pandas as pd # for data manipulation
import numpy as np # for numerical operations
import mne # for EEG/MEG data processing
import os # for file operations
import seaborn as sns # for data visualization

# Quick analysis of EEG data for pilot participant in functional localizer task (MP - June 10 2025)

# Load the data
participant = 'sub-088'

# one level up to the sourcedata folder
data_path = f"../sourcedata/{participant}"

# Find the eeg vhdr file
eeg_files = [f for f in os.listdir(data_path) if '.vhdr' in f]

# Read the EEG data and specify channel types
raw = mne.io.read_raw(os.path.join(data_path, eeg_files[0]), eog=['HEOG', 'VEOG'], misc=['ECG'] , preload=True)

# Load the behavioral data
behav_file = pd.read_csv("../sourcedata/sub-088/func_loc/sub-088_FunctionalLocalizer_2025-06-10_15h24.05.961.csv")

# Drop the rows that are not trials (no image_file)
behav_file = behav_file.dropna(subset=['image_file'])


# Set the montage for the EEG data
raw.set_montage('easycap-M1')

# PLot the EEG montage
raw.plot_sensors(show_names=True)

# Ge the events from the raw data
# Plot the events
events, event_id = mne.events_from_annotations(raw)

# Crop at the last event + 5 s 
last_event_time = events[-1, 0] / raw.info['sfreq']  # Convert to seconds
raw.crop(tmax=last_event_time + 5)

# Drop non EEG channels
raw.drop_channels(['HEOG', 'VEOG', 'ECG'])

# Filter 
raw.filter(1, 40)

# Plot the psd
raw.plot_psd(fmax=60)

# Plot the events to visualize them
mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info['sfreq'], first_samp=raw.first_time)

# Keep only the stimuli and the following photosensor
events = events[(events[:, 2] == event_id['Stimulis/S  1']) | (events[:, 2] == event_id['Photosensor/P  1'])]


diff_time = np.diff(events[:, 0])

# Invert event_id dictionary to map event codes to names
event_id_inv = {v: k for k, v in event_id.items()}

frame_diff = pd.DataFrame(dict(
    frame_diff=diff_time,
    frame=events[1:, 0],
    event_type=events[1:, 2],
))

frame_diff['event_name'] = frame_diff['event_type'].map(event_id_inv)

# Check distribution of time differences for event Photosensor/P  1
frame_diff_photo = frame_diff[frame_diff['event_name'] == 'Photosensor/P  1']
sns.histplot(frame_diff_photo['frame_diff'], bins=100, kde=True)

# Keep only the stimuli events
events_stimuli = events[events[:, 2] == event_id['Stimulis/S  1']]

# Shift the events by ~9 ms to align with the photosensor
events_stimuli[:, 0] += 9




# Make sure we have the same number of events as trials
if len(events_stimuli) != len(behav_file):
    raise ValueError("Number of events does not match number of trials in behavioral data.")

# Create epochs around the stimuli events
epochs = mne.Epochs(raw, events_stimuli, tmin=-0.2, tmax=1, baseline=(-0.2, 0), preload=True)

# Add the behavioral data to the epochs
epochs.metadata = behav_file


# Calculate performance
epochs.metadata['response_correct'].mean()

print(f"Mean response accuracy: {epochs.metadata['response_correct'].mean():.2f}")

# Drop the incorrect trials from the epochs
epochs_correct = epochs[epochs.metadata['response_correct'] == 1]

# Drop very bad trials (e.g., those with very high amplitude)

# PLot the erp image for the correct trials
epochs_correct.plot_image(picks='eeg', combine='mean', vmin=-10, vmax=10, cmap='RdBu_r')


# PLot the erp image for the incorrect trials
epochs_incorrect = epochs[epochs.metadata['response_correct'] == 0]
epochs_incorrect.plot_image(picks='eeg', combine='mean', vmin=-10, vmax=10, cmap='RdBu_r')


# Compare evoked for each image_file
image_files = epochs.metadata['image_file'].unique()
all_evokeds = []
for image_file in image_files:
    epochs_image = epochs[epochs.metadata['image_file'] == image_file]
    evoked_image = epochs_image.average()
    all_evokeds.append(evoked_image)
# Plot the evoked responses for each image
# Replace legend labels with image file names
for i, evoked in enumerate(all_evokeds):
    evoked.comment = image_files[i]
mne.viz.plot_compare_evokeds(all_evokeds, picks='eeg', combine='mean', ci=0.95, show_sensors='upper right')


# Do some decoding analysis using a simple logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer, Scaler
# Create a pipeline with a scaler and a logistic regression classifier
pipeline = make_pipeline(
    Vectorizer(),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
# Prepare the data for decoding
X = epochs_correct.get_data()
y = epochs_correct.metadata['image_file'].values
# Perform cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print(f"Decoding accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# box plot of the decoding accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.boxplot(y=scores)
plt.title('Decoding Accuracy')
plt.ylabel('Accuracy')
plt.axhline(1/len(np.unique(y)), color='red', linestyle='--', label='Chance Accuracy')
plt.legend
plt.show()

# try with a sliding window approach
from mne.decoding import SlidingEstimator, cross_val_multiscore

clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

time_decod = SlidingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)
# here we use cv=3 just for speed
scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=None)


# Fit the sliding estimator with cross-validation


# Plot the sliding window scores
plt.figure(figsize=(10, 6))
plt.plot(epochs_correct.times, scores.mean(axis=0), label='Sliding Window Accuracy')
plt.axhline(1/len(np.unique(y)), color='red', linestyle='--', label='Chance Accuracy')
plt.axvline(0, color='black', linestyle='--', label='Time 0 (Stimulus Onset)')
plt.fill_between(epochs_correct.times, 
                 scores.mean(axis=0) - scores.std(axis=0), 
                 scores.mean(axis=0) + scores.std(axis=0), 
                 alpha=0.3, label='±1 Std Dev')
plt.title('Sliding Window Decoding Accuracy')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()