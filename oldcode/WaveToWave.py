import mido
from mido import Message
import time

# List available MIDI output ports
print("Available MIDI output ports:")
output_ports = mido.get_output_names()
print(output_ports)

# Open the correct port (replace name if needed)
outport = mido.open_output('PythonPort 1')

print("Connected to MIDI port. Sending 4 notes, 5 seconds apart...")

# Define 4 notes to send
notes = [60, 62, 64, 65]  # C4, D4, E4, F4

for i, note in enumerate(notes):
    print(f"Sending note {note} (index {i})")
    outport.send(Message('note_on', note=note, velocity=100))
    time.sleep(1)  # keep note on for 1 second
    outport.send(Message('note_off', note=note, velocity=0))
    if i < len(notes) - 1:
        time.sleep(5)  # wait 5 seconds before next note

print("Finished sending notes.")
outport.close()
