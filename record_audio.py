import pyaudio
import wave

# set the chunk size and recording format
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1

# initialize the PyAudio object
audio = pyaudio.PyAudio()

# open the microphone stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=chunk)

# create a buffer to hold the audio data
frames = []

# record audio in chunks and append to buffer
for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
    data = stream.read(chunk)
    frames.append(data)

# stop the microphone stream
stream.stop_stream()
stream.close()

# terminate the PyAudio object
audio.terminate()

# save the audio data to a file
wf = wave.open("output.wav", "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()
