
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

EPOCHS = 30
BATCH_SIZE = 50

# DATASET
DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
		data_dir,
		batch_size=BATCH_SIZE,
		validation_split=0.2,
		seed=0,
		output_sequence_length=16000,
		subset='both'
)

label_names = np.array(train_ds.class_names)
print('labels:', label_names)

def squeeze(audio, labels):
	audio = tf.squeeze(audio, axis=-1)
	return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

def get_spectrogram(waveform):
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	spectrogram = spectrogram[..., tf.newaxis]
	return spectrogram

def make_spec_ds(ds):
	return ds.map(
		map_func=lambda audio,
		label: (
			get_spectrogram(audio),
			label
		),
		num_parallel_calls=tf.data.AUTOTUNE
	)
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
	break

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = tf.keras.Sequential([
	layers.Input(shape=input_shape),
	# Downsample the input.
	layers.Resizing(32, 32),
	# Normalize.
	norm_layer,
	layers.Conv2D(32, 3, activation='relu'),
	layers.Conv2D(64, 3, activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.25),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dropout(0.5),
	layers.Dense(num_labels),
])

model.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'],
)

history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

model.evaluate(test_spectrogram_ds, return_dict=True)

model.save('model2.keras')