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
import csv
import tensorflow_hub as hub

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile

from helpers.readlabels import readLabels
from helpers.audio import *
from helpers.model import *

MODEL = 'model5'
REALTIME_DIR = 'data/realtime'
CHUPLENGTH = 10
CONFIDENCE_RATE = .9999

label_names = np.array(readLabels(MODEL))

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
    return output_data.getvalue()

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

    def ensure_sample_rate(original_sample_rate, waveform,
                        desired_sample_rate=16000):
        if original_sample_rate != desired_sample_rate:
            desired_length = int(round(float(len(waveform)) /
                                    original_sample_rate * desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform

    wav_file_name = './yamnet/speech_whistling2.wav'
    # wav_file_name = './yamnet/miaow_16k.wav'


    output_stream = io.BytesIO(audiotopredict)
    # Read the WAV file from the BytesIO object
    sample_rate, wav_data = wavfile.read(output_stream, 'rb')

    # sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Show some basic information about the audio.
    duration = len(wav_data)/sample_rate
    # print(f'Sample rate: {sample_rate} Hz')
    # print(f'Total duration: {duration:.2f}s')
    # print(f'Size of the input: {len(wav_data)}')

    # Listening to the wav file.
    Audio(wav_data, rate=sample_rate)

    waveform = wav_data / tf.int16.max

    # Run the model, check the output.
    scores, embeddings, spectrogram = checkmodel(waveform)

    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    # print(f'The main sound is: {infered_class}')
    print(infered_class)
    return infered_class

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

    # prepare predictions
    if isitspeech(parameter) == "Speech":
        prediction = model(processaudio(audiodata=parameter, address=''))
        predictions = list(tf.nn.softmax(prediction[0]).numpy())
        
        # Diaply statistics(predictions)
        stat(predictions)
        # action(predictions)
        
    