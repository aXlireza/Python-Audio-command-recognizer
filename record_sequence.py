import tensorflow as tf
import pyaudio
import wave
import numpy as np

data_dir = 'data/custom_commands/ava'
count = 112

# Set up the microphone
chunk = 1024 # Record in chunks of 1024 samples
format = pyaudio.paInt16 # 16-bit resolution
channels = 1 # Mono
rate = 16000 # Sample rate
record_seconds = 1 # Record for 1 second

p = pyaudio.PyAudio() # Create an instance of PyAudio

stream = p.open(
    format=format,
    channels=channels,
    rate=rate,
    input=True,
    frames_per_buffer=chunk
)

# Implement the loop
while True:
    # Record the audio
    frames = []
    print(str(count) + "Recording....")
    for i in range(0, int(rate/chunk*record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Done!!")
    
    wav = wave.open(data_dir+'/'+str(count)+'.wav', 'wb')
    # wav = wave.open('data/realtime/temp.wav', 'wb')
    wav.setnchannels(channels)
    wav.setsampwidth(p.get_sample_size(format))
    wav.setframerate(rate)
    wav.writeframes(b''.join(frames))
    wav.close()
    count += 1