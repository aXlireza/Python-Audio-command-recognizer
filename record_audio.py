import pyaudio
from helpers.audio import *

# set the chunk size and recording format
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1

audio = pyaudio.PyAudio()
stream = audiostream(audio, FORMAT, CHANNELS, RATE, chunk)
record_audio(audio, stream, int(RATE / chunk * RECORD_SECONDS), FORMAT, CHANNELS, RATE, chunk, "output.wav", True)
terminateaudio(audio, stream)
