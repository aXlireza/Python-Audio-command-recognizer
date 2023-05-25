import os

# Specify the path to your Chrome executable and the profile directory
chrome_path = '/usr/bin/google-chrome'
profile_directory = '~/.config/google-chrome/Profile 3'

# Command to open the ChatGPT website with the specified profile
# os.system(f'{chrome_path} --profile-directory="{profile_directory}" https://chat.openai.com/')
os.system(f'google-chrome-stable https://chat.openai.com/')