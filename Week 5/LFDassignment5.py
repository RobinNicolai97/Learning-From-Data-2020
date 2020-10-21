#!/usr/bin/env python3
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils, generic_utils
np.random.seed(2018)  # for reproducibility and comparability, don't change!
import json
from sklearn.preprocessing import label_binarize

def get_embedding(word, embeddings):
	try:
		# GloVe embeddings only have lower case words
		return embeddings[word.lower()]
	except KeyError:
		return embeddings['UNK']

# Load noun-noun compound data
def load_data():
	print("Loading data...")
	# Embeddings
	embeddings = json.load(open('embeddings.json', 'r'))
	# Training and development data
	X_train = []
	Y_train = []
	with open('training_data.tsv', 'r') as f:
		for line in f:
			split = line.strip().split('\t')
			# Get feature representation
			embedding_1 = get_embedding(split[0], embeddings)
			embedding_2 = get_embedding(split[1], embeddings)
			X_train.append(embedding_1 + embedding_2)
			# Get label
			label = split[2]
			Y_train.append(label)
	classes = sorted(list(set(Y_train)))
	X_train = np.array(X_train)
	# Convert string labels to one-hot vectors
	Y_train = label_binarize(Y_train, classes)
	Y_train = np.array(Y_train)
	# Split off development set from training data
	X_dev = X_train[-3066:]
	Y_dev = Y_train[-3066:]
	X_train = X_train[:-3066]
	Y_train = Y_train[:-3066]
	print(len(X_train), 'training instances')
	print(len(X_dev), 'develoment instances')
	# Test data
	X_test = []
	Y_test = []
	with open('test_data_clean.tsv', 'r') as f:
		for line in f:
			split = line.strip().split('\t')
			# Get feature representation
			embedding_1 = get_embedding(split[0], embeddings)
			embedding_2 = get_embedding(split[1], embeddings)
			X_test.append(embedding_1 + embedding_2)
	X_test = np.array(X_test)
	print(len(X_test), 'test instances')

	return X_train, X_dev, X_test, Y_train, Y_dev, classes

# Build confusion matrix with matplotlib	
def create_confusion_matrix(true, pred):	
	import matplotlib.pyplot as plt
	from sklearn.metrics import confusion_matrix
	# Build matrix
	cm = confusion_matrix(true, pred, labels = classes)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	# Make plot
	plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Blues)
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.xlabel('Predicted label')
	plt.yticks(tick_marks, classes)
	plt.ylabel('True label')
	plt.show()
	
if __name__ == '__main__':
	# Read arguments
	parser = argparse.ArgumentParser(description='NN parameters')
	parser.add_argument('-r', '--run', type=int, default=0, help='run number')
	parser.add_argument('-e', '--epochs', metavar='N', type=int, default=5, help='epochs')
	parser.add_argument('-bs', '--batch-size', metavar='N', type=int, default=50, help='batch size')
	parser.add_argument('-cm', '--confusion-matrix', action='store_true', help = 'Show confusion matrix. Requires matplotlib')
	args = parser.parse_args()
	
	# Load data
	X_train, X_dev, X_test, Y_train, Y_dev, classes = load_data()
	nb_features = X_train.shape[1]
	print(nb_features, 'features')
	nb_classes = Y_train.shape[1]
	print(nb_classes, 'classes')
	
	# Build the model 
	print("Building model...")
	model = Sequential()
	# Single 500-neuron hidden layer with sigmoid activation
	model.add(Dense(input_dim = nb_features, units = 500, activation = 'relu'))
	# Output layer with softmax activation
	model.add(Dense(units = nb_classes, activation = 'softmax'))
	# Specify optimizer, loss and validation metric
	model.compile(loss='KLDivergence', optimizer='sgd', metrics=['accuracy'])
	# Train the model 
	history = model.fit(X_train, Y_train, epochs = args.epochs, batch_size = args.batch_size, validation_data = (X_dev, Y_dev), shuffle = True, verbose = 1)

	# Predict labels for test set
	outputs = model.predict(X_test, batch_size=args.batch_size)
	pred_classes = np.argmax(outputs, axis=1)
	
	# Save predictions to file
	np.save('test_set_predictions_run{0}'.format(args.run), pred_classes)

	# Make confusion matrix on development data
	if args.confusion_matrix:
		Y_dev_names = [classes[x] for x in np.argmax(Y_dev, axis=1)]
		pred_dev = model.predict(X_dev, batch_size = args.batch_size)
		pred_class_names = [classes[x] for x in np.argmax(pred_dev, axis = 1)]
		create_confusion_matrix(Y_dev_names, pred_class_names)
