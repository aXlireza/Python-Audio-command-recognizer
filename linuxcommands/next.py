import os

# Check if music is playing using 'mpc status' command
status = os.system("pacmd list-sink-inputs | grep -q 'state: RUNNING' && exit 0 || exit 1")
if status == 0:
    # Music is playing, so play the next song using 'mpc next' command
    os.system("xdotool key XF86AudioNext")
else:
    # Music is not playing, simulate a right-click using 'xdotool' command
    os.system('xdotool key Right')