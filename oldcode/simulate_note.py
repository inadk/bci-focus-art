import mido
import time
import random
from mido import Message

# --- MIDI SETUP ---
print("Available MIDI output ports:", mido.get_output_names())
midi_port = 'PythonPort 1'  # Change to your actual MIDI port
outport = mido.open_output(midi_port)
print(f"Using MIDI port: {midi_port}")

# --- SETTINGS ---
note_blink = 60  # Middle C (for blink simulation)
note_left = 64   # Left hemisphere note
note_right = 62  # Right hemisphere note
note_neutral = 65  # Neutral hemisphere note

# --- Timers ---
last_hem_time = time.time()  # Hemisphere comparison every 3 seconds

# Variable to keep track of the last hemisphere activity
last_hem_side = None
next_blink_time = time.time() + random.uniform(0, 8)  # Random start time for the first blink

try:
    while True:
        # --- Simulate Blink at Random Intervals (0 to 8 seconds) ---
        if time.time() >= next_blink_time:
            print(f"[{time.strftime('%H:%M:%S')}] Simulated blink!")
            
            # Send MIDI note on (blink)
            outport.send(Message('note_on', note=note_blink, velocity=100))
            time.sleep(0.1)  # Keep the note on for a short moment
            
            # Send MIDI note off (end blink)
            outport.send(Message('note_off', note=note_blink, velocity=0))
            
            # Set the next blink time to be a random time between 0 and 8 seconds
            next_blink_time = time.time() + random.uniform(0, 8)

        # --- Simulate Hemisphere Comparison Every 3 Seconds ---
        if time.time() - last_hem_time >= 3:
            # Simulating random hemisphere comparison values
            side = random.choice(["Left - Happy", "Right - Sad", "Neutral"])
            
            # Only send a message if the hemisphere side is different from the last one
            if side != last_hem_side:
                if side == "Left - Happy":
                    note = note_left
                elif side == "Right - Sad":
                    note = note_right
                else:
                    note = note_neutral

                print(f"[{time.strftime('%H:%M:%S')}] {side} - indicated by hemisphere activity")

                # Send MIDI note for hemisphere comparison
                outport.send(Message('note_on', note=note, velocity=100))
                time.sleep(0.1)  # Short note duration

                # Send MIDI note off (end hemisphere comparison)
                outport.send(Message('note_off', note=note, velocity=0))

                # Update the hemisphere activity state
                last_hem_side = side

            # Update the hemisphere comparison time
            last_hem_time = time.time()

except KeyboardInterrupt:
    print("Simulation stopped by user.")

finally:
    outport.close()
    print("MIDI port closed.")
