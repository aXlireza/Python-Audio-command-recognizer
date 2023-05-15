import tensorflow as tf
import pyaudio
import wave
import io
import audioop
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import subprocess
import os

from helpers.readlabels import readLabels
from helpers.audio import *

MODEL = 'model4'
REALTIME_DIR = 'data/realtime'
CHUPLENGTH = 10
CONFIDENCE_RATE = .9999
realtime_data_dir = pathlib.Path(REALTIME_DIR)
data_dir = pathlib.Path('data/mini_speech_commands')

label_names = np.array(readLabels(MODEL))

def get_spectrogram(waveform):
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	spectrogram = spectrogram[..., tf.newaxis]
	return spectrogram

# Load your trained TensorFlow model
model = tf.keras.models.load_model('models/'+MODEL+'/model.keras')

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

    # remove off top
    recordedfiles = sorted(os.listdir(REALTIME_DIR), key=lambda x: int(x.split('.')[0]))
    num_files = len(recordedfiles)
    os.remove(os.path.join(REALTIME_DIR, recordedfiles[0])) if num_files == CHUPLENGTH+1 else ""

def read_record():
    # Sort the files list numerically
    files_sorted = sorted(os.listdir(REALTIME_DIR), key=lambda x: int(x.split('.')[0]))

    # Open the first file to get the parameters
    with wave.open(os.path.join(REALTIME_DIR, files_sorted[0]), 'rb') as first_file:
        params = first_file.getparams()

        # Create a BytesIO object to hold the output data
        output_data = io.BytesIO()

        # Create the output file
        with wave.open(output_data, 'wb') as output_file:

            # Write the parameters to the output file
            output_file.setparams(params)

            # Write the data from each file to the output file
            for filename in files_sorted:
                with wave.open(os.path.join(REALTIME_DIR, filename), 'rb') as input_file:
                    output_file.writeframes(input_file.readframes(input_file.getnframes()))

    # Get the output data as a bytes object
    x = output_data.getvalue()

    # my_plot_data = realtime_data_dir/'temp.wav'
    # x = tf.io.read_file(str(my_plot_data))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    return x

def render(id):
    print(label_names[id], ":", predictions[id])

def stat(predictions):
    sorted_indices = np.argsort(predictions)[::-1]
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
    if sorted_indices[0] == 14 and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/mute.py"])
    # # up => volume up
    if sorted_indices[0] == 29 and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/volumeup.py"])
    # # down => volume down
    if sorted_indices[0] == 4 and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/volumedown.py"])
    # # stop => lock
    if sorted_indices[0] == 25 and predictions[sorted_indices[0]] >= CONFIDENCE_RATE : subprocess.run(["python", "./linuxcommands/lockscreen.py"])

# Loop through the list and delete each file
for file_name in os.listdir(REALTIME_DIR):
    file_path = os.path.join(REALTIME_DIR, file_name)
    os.remove(file_path)

# Implement the loop
while True:
    # Record the audio
    record()

    # predict
    parameter = read_record()
    prediction = model(parameter)

    # prepare predictions
    predictions = list(tf.nn.softmax(prediction[0]).numpy())
    
    # Diaply statistics(predictions)
    stat(predictions)
    action(predictions)
    
    