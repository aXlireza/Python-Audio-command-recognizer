import tensorflow as tf
import matplotlib.pyplot as plt
from helpers.audio import *

def readmodel(name):
	return tf.keras.models.load_model('models/'+name+'/model.keras')

def compile(model, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']):
	model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

def prediction_bar(model, label_names, x, name='no'):
	predictions = model(processaudio(address=x, audiodata=''))
	print(predictions[0])
	plt.bar(label_names, tf.nn.softmax(predictions[0]))
	plt.title(name)
	plt.show()
