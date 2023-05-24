import wave
import tensorflow as tf
import os
import io

REALTIME_DIR = 'data/realtime'


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

def get_spectrogram(waveform):
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	spectrogram = spectrogram[..., tf.newaxis]
	return spectrogram

def processaudio(address, audiodata):
    x = tf.io.read_file(str(address)) if address else audiodata
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    return x

def read_record():
    # Sort the files list numerically
    files_sorted = sorted(os.listdir(REALTIME_DIR), key=lambda x: int(x.split('.')[0]))

    # Open the first file to get the parameters
    with wave.open(os.path.join(REALTIME_DIR, files_sorted[0]), 'rb') as first_file:
        params = first_file.getparams()
        # params = wave.Wave_write.setparams(
        #     nchannels=1,
        #     sampwidth=format,
        #     framerate=16000,
        #     nframes=16000 * 1,
        #     comptype="NONE",
        #     compname="not compressed"
        # )
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
