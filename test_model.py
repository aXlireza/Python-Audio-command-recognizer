import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# DATASET
DATASET_PATH = 'data/mini_speech_commands'
CUSTOM_DATASET_PATH = 'data/custom'

data_dir = pathlib.Path(DATASET_PATH)
custom_data_dir = pathlib.Path(CUSTOM_DATASET_PATH)

label_names = np.array(['down', 'go', 'hello', 'left', 'mute', 'no', 'right', 'sasho', 'stop', 'up', 'yes'])

def get_spectrogram(waveform):
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	spectrogram = spectrogram[..., tf.newaxis]
	return spectrogram

model = tf.keras.models.load_model('model2.keras')


model.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'],
)

def prediction_bar(x, name='no'):
	x = tf.io.read_file(str(x))
	x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
	x = tf.squeeze(x, axis=-1)
	x = get_spectrogram(x)
	x = x[tf.newaxis,...]

	prediction = model(x)
	print(prediction[0])
	plt.bar(label_names, tf.nn.softmax(prediction[0]))
	plt.title(name)
	plt.show()

# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
# for x in commands:
# 	prediction_bar(data_dir/x, x)
prediction_bar(data_dir/'sasho/1.wav', 'sasho')
prediction_bar(data_dir/'sasho/5.wav', 'sasho')
prediction_bar(data_dir/'sasho/12.wav', 'sasho')
prediction_bar(data_dir/'sasho/20.wav', 'sasho')
