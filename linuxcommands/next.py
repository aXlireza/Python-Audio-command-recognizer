import os

# Check if music is playing using 'mpc status' command
status = os.system('mpc status > /dev/null 2>&1')
if status == 0:
    # Music is playing, so play the next song using 'mpc next' command
    os.system('mpc next')
else:
    # Music is not playing, simulate a right-click using 'xdotool' command
    os.system('xdotool click 3')