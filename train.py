import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from helpers.audio import *

# Set the seed value for experiment reproducibility.
SEED = 42
EPOCHS = 120
BATCH_SIZE = 27
VALIDATION_SPLIT = 0.2
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_SAVE='models/AVA3/model.keras'
DATASET_PATH = 'data/custom_commands'
data_dir = pathlib.Path(DATASET_PATH)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    data_dir,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    seed=SEED,
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

def make_spec_ds(ds):
	return ds.map(
		map_func=lambda audio,
		label: (get_spectrogram(audio), label),
		num_parallel_calls=tf.data.AUTOTUNE
	)
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1): break

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

norm_layer = layers.BatchNormalization()

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
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.TimeDistributed(layers.Flatten()),
    layers.GRU(128, return_sequences=True),
    layers.Dropout(0.5),
    layers.GRU(128),
    layers.Dropout(0.5),
    layers.Dense(num_labels, activation='softmax'),
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
    # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
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

model.save(MODEL_SAVE)