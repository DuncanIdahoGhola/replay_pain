from psychopy import visual, core, event
import math # We'll use math.ceil for a more natural countdown

# --- Experiment Setup ---
# Create a window
win = visual.Window(
    size=[800, 600],  # Window size in pixels
    color="lightgray", # Background color
    units="pix"       # Use pixels for units
)

# --- Timer Setup ---
countdown_duration = 60  # seconds

# Create a text stimulus to display the timer
timer_text_stim = visual.TextStim(
    win,
    text=str(countdown_duration), # Initial text
    pos=(0, 0),                  # Centered
    color="black",
    height=50                    # Text height in pixels
)

# Create a clock for the countdown
countdown_clock = core.Clock()

# --- Main Experiment Loop ---
keep_running = True
countdown_clock.reset() # Start the clock from zero just before the loop

while keep_running:
    elapsed_time = countdown_clock.getTime()
    remaining_time = countdown_duration - elapsed_time

    # --- Version 1: Simple Seconds Display (e.g., 60, 59, ..., 0) ---
    # We use math.ceil to make it show "60" for the first full second,
    # then "59" for the next, etc.
    # If you prefer it to immediately show "59" once the first second starts,
    # you could use int() or math.floor().
    if remaining_time > 0:
        display_seconds = math.ceil(remaining_time)
    else:
        display_seconds = 0
    timer_text_stim.setText(str(display_seconds))

    # --- Version 2: MM:SS Format Display (e.g., 01:00, 00:59, ..., 00:00) ---
    # Uncomment this section and comment out Version 1 if you prefer this format.
    """
    if remaining_time > 0:
        minutes = int(remaining_time // 60)
        seconds = math.ceil(remaining_time % 60)
        # Handle the case where ceil(seconds % 60) becomes 60 (e.g., at exactly 60.0 seconds)
        if seconds == 60 and minutes < (countdown_duration // 60) : # if not the very start
            minutes +=1
            seconds = 0
        # Ensure seconds don't exceed 59 in display when minutes are > 0
        if minutes > 0 and seconds == 60: # e.g. if remaining time is exactly 120.0, it becomes 02:00 not 01:60
            seconds = 0
    else:
        minutes = 0
        seconds = 0
    timer_text_stim.setText(f"{minutes:02d}:{seconds:02d}") # :02d pads with leading zero
    """


    # --- Draw stimuli ---
    timer_text_stim.draw()
    win.flip() # Update the screen

    # --- Check for quit (Escape key) ---
    keys = event.getKeys(keyList=['escape'])
    if 'escape' in keys:
        keep_running = False

    # --- End condition for the timer ---
    if remaining_time <= 0:
        # Timer has finished
        # Optionally, you can make it stay at "0" or "00:00" for a bit
        # For now, we'll just end the loop.
        # To show "0" for a brief moment before exiting the loop:
        if remaining_time < -0.5: # e.g., allow 0.5s to display "0"
           keep_running = False
        # If you want it to stop immediately when it hits 0:
        # keep_running = False


# --- End of Experiment ---
# Optional: Display a "Time's Up!" message
if not ('escape' in keys): # Only show if not manually escaped
    time_up_message = visual.TextStim(win, text="Time's Up!", color="red", height=60)
    time_up_message.draw()
    win.flip()
    core.wait(2) # Display for 2 seconds

win.close()
core.quit()