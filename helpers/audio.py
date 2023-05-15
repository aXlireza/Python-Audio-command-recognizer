import wave

def audiostream(audio, format, channels, rate, chunk):
    return audio.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk
    )

def terminateaudio(audio, stream):
    # stop the microphone stream
    stream.stop_stream()
    stream.close()

    # terminate the PyAudio object
    audio.terminate()

def record_audio(audio, stream, length, format, channels, rate, chunk, address, dev):
    frames = []
    print("Recording....") if dev else ""
    for i in range(0, length):
        data = stream.read(chunk)
        frames.append(data)
    print("Done!!") if dev else ""
    # Process the audio
    wav = wave.open(address, 'wb')
    wav.setnchannels(channels)
    wav.setsampwidth(audio.get_sample_size(format))
    wav.setframerate(rate)
    wav.writeframes(b''.join(frames))
    wav.close()
