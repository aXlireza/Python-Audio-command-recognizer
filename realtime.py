import tensorflow as tf
import pyaudio
import wave
import numpy as np
import subprocess
import os
import csv
import tensorflow_hub as hub
from scipy.io import wavfile
from helpers.readlabels import readLabels
from helpers.audio import *
from helpers.model import *
import time


MODEL = 'AVA4'
REALTIME_DIR = 'data/realtime'
CHUPLENGTH = 3
CONFIDENCE_RATE = .1099

label_names = np.array(readLabels(MODEL))

# CoolDown mechanism
cooldown_duration = 2  # Set the cooldown duration in seconds
last_detection_time = 0

# Load your trained TensorFlow model
model = readmodel(MODEL)
# Load the model to check if it's speech
checkmodel = hub.load('./yamnet/yamnet_1')

# Set up the microphone
chunk = 1024 # Record in chunks of 1024 samples
format = pyaudio.paInt16 # 16-bit resolution
channels = 1 # Mono
rate = 16000 # Sample rate
record_seconds = 1 # Record for 1 second
recorded_file_count = 0

p = pyaudio.PyAudio() # Create an instance of PyAudio
stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

def record():
    global recorded_file_count
    record_audio(p, stream, int(rate/chunk/CHUPLENGTH), format, channels, rate, chunk, REALTIME_DIR+'/'+str(recorded_file_count)+'.wav', False)
    recorded_file_count += 1

    recordedfiles = sorted(os.listdir(REALTIME_DIR), key=lambda x: int(x.split('.')[0]))
    num_files = len(recordedfiles)
     # Create the output file
    with wave.open(os.path.join(REALTIME_DIR, recordedfiles[0]), 'rb') as first_file:
        params = first_file.getparams()
        with wave.open('data/realtime.wav', 'wb') as output_file:
            # Write the parameters to the output file
            output_file.setparams(params)

            # Write the data from each file to the output file
            for filename in recordedfiles:
                with wave.open(os.path.join(REALTIME_DIR, filename), 'rb') as input_file:
                    output_file.writeframes(input_file.readframes(input_file.getnframes()))
    # remove off top
    os.remove(os.path.join(REALTIME_DIR, recordedfiles[0])) if num_files == CHUPLENGTH+1 else ""

def render(id):
    print(label_names[id], ":", predictions[id])

def stat(predictions, sorted_indices):
    top_three_values = sorted_indices[:3]
    
    # # # print topm three predictions
    # for index in top_three_values: render(index)
    # print("==================================")

    # # # Print the top prediction
    # render(top_three_values[0])

    # # # only print hight confident one
    if predictions[top_three_values[0]] >= CONFIDENCE_RATE: render(top_three_values[0])

def action(predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    # # mute
    if label_names[sorted_indices[0]] == "mute" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/mute.py"])
    # # unmute
    if label_names[sorted_indices[0]] == "unmute" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/unmute.py"])
    # # up => volume up
    if label_names[sorted_indices[0]] == "up" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/volumeup.py"])
    # # down => volume down
    if label_names[sorted_indices[0]] == "down" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/volumedown.py"])
    # # stop => lock
    if label_names[sorted_indices[0]] == "stop" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/lockscreen.py"])
    # # gpt => pull up chat gpt
    # if label_names[sorted_indices[0]] == "gpt" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/openchatgpt.py"])
    # # play
    if label_names[sorted_indices[0]] == "play" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/play.py"])
    # # pause
    if label_names[sorted_indices[0]] == "pause" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/pause.py"])
    # # music
    if label_names[sorted_indices[0]] == "music" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/music.py"])
    # # next
    if label_names[sorted_indices[0]] == "next" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/next.py"])
    # # server
    # if label_names[sorted_indices[0]] == "server" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/pause.py"])

def isitspeech(audiotopredict):

    # Find the name of the class with the top score when mean-aggregated across frames.
    def class_names_from_csv(class_map_csv_text):
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader: class_names.append(row['display_name'])
        return class_names

    class_map_path = checkmodel.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
        if original_sample_rate != desired_sample_rate:
            desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform

    sample_rate, wav_data = wavfile.read(audiotopredict, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    waveform = wav_data / tf.int16.max
    scores, embeddings, spectrogram = checkmodel(waveform)
    scores_np = scores.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    return infered_class

# Loop through the list and delete each file
for file_name in os.listdir(REALTIME_DIR):
    file_path = os.path.join(REALTIME_DIR, file_name)
    os.remove(file_path)

def process_command(predictions, sorted_indices):
    # Diaply statistics(predictions)
    stat(predictions, sorted_indices)
    action(predictions)

def cooldown(current_time):
    time_since_last_detection = current_time - last_detection_time
    return time_since_last_detection

actionable = False

def is_music_playing():
    try:
        output = subprocess.check_output(["playerctl", "status"]).decode("utf-8").strip()
        if output == "Playing":
            return True
    except subprocess.CalledProcessError:
        pass

    return False

wasitplaying = None
secondary_wasitplaying = None
actionable = False

while True:
    record()

    if isitspeech('data/realtime.wav') == "Speech":
        # COMMAND RECOGNITION
        prediction = model(processaudio(audiodata='', address='data/realtime.wav'))
        predictions = list(tf.nn.softmax(prediction[0]).numpy())
        sorted_indices = np.argsort(predictions)[::-1]
        if label_names[sorted_indices[0]] != "noise":
            
            current_time = time.time()
            cooldown(current_time)

            if actionable == False and label_names[sorted_indices[0]] == "ava" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE and cooldown(current_time) > cooldown_duration:
                print("ACTIVATED")
                wasitplaying = is_music_playing()
                actionable = True
                if wasitplaying == True: subprocess.run(["python", "./linuxcommands/pause.py"])
            elif actionable == True and label_names[sorted_indices[0]] != "ava" and predictions[sorted_indices[0]] >= CONFIDENCE_RATE and cooldown(current_time) > cooldown_duration:
                actionable = False
                last_detection_time = current_time
                if wasitplaying == True: subprocess.run(["python", "./linuxcommands/play.py"])
                process_command(predictions, sorted_indices)
                print("DEACTIVATED")