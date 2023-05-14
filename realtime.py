import tensorflow as tf
import pyaudio
import wave
import audioop
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import subprocess

from helpers.readlabels import readLabels

MODEL = 'model2'
realtime_data_dir = pathlib.Path('data/realtime')
data_dir = pathlib.Path('data/mini_speech_commands')

label_names = np.array(readLabels('model2'))
# label_names = np.array(['down', 'go', 'hello', 'left', 'mute', 'no', 'right', 'sasho', 'stop', 'up', 'yes'])
# label_names = np.array(['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy'
#  'hello', 'house', 'left', 'marvel', 'mute', 'nine', 'no', 'off', 'on', 'one'
#  'right', 'sasho', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up'
#  'wow', 'yes', 'zero'])

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

p = pyaudio.PyAudio() # Create an instance of PyAudio
stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

def record(record_seconds = 1):
    frames = []
    print("Recording....")
    for i in range(0, int(rate/chunk*record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Done!!")
    # Process the audio
    wav = wave.open('data/realtime/temp.wav', 'wb')
    wav.setnchannels(channels)
    wav.setsampwidth(p.get_sample_size(format))
    wav.setframerate(rate)
    wav.writeframes(b''.join(frames))
    wav.close()

def read_record():
    my_plot_data = realtime_data_dir/'temp.wav'
    x = tf.io.read_file(str(my_plot_data))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    return x

def stat(predictions, hightest_confidence_index):
    def render(id):
        print(label_names[id], ":", predictions[id], "+" if hightest_confidence_index==id else "")
    for n in range(len(label_names)): render(n)
    print("==================================")
    

# Implement the loop
while True:
    # Record the audio
    record()

    # predict
    prediction = model(read_record())

    # prepare predictions
    predictions = list(tf.nn.softmax(prediction[0]).numpy())
    hightest_confidence_index = np.argmax(prediction, axis=1)
    # Diaply statistics(predictions)
    stat(predictions, hightest_confidence_index)
    
    # # mute
    # if hightest_confidence_index[0] == 4 : subprocess.run(["python", "./linuxcommands/mute.py"])
    # # stop => lock
    # if hightest_confidence_index[0] == 8 : subprocess.run(["python", "./linuxcommands/lockscreen.py"])