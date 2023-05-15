import tensorflow as tf
import matplotlib.pyplot as plt

def get_spectrogram(waveform):
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	spectrogram = spectrogram[..., tf.newaxis]
	return spectrogram

def readmodel(name):
	return tf.keras.models.load_model('models/'+name+'/model.keras')

def compile(model, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']):
	model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

def prediction_bar(model, label_names, x, name='no'):
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
