# surpress warnings and information
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import
import sys
import json
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

STOPWORDS = set(stopwords.words('english'))


# Read all possible files in a folder
def read_corpus(dir_name='data'):
	print("Reading files")

	file_list = os.listdir(dir_name)
	if '.DS_Store' in file_list:
		file_list.remove('.DS_Store')

	article_list = []
	label_list = []

	for file in file_list:
		with open(dir_name + "/" + file) as f:
			data = json.load(f)
			articles = data['articles']
			for article in articles:
				article_list.append(article['body'])
				# Retrieve labels
				subjects = article['classification']['subject']
				if subjects:
					labels = []
					for label in subjects:
						labels.append(label['name'])
					label_list.append(labels)
				else:
					label_list.append('ZNONE')

	return article_list, label_list


# Read a single file
def read_file(file_name):
	article_list = []
	label_list = []

	with open(file_name) as f:
		data = json.load(f)
		articles = data['articles']
		for article in articles:
			article_list.append(article['body'])
			# Retrieve labels
			subjects = article['classification']['subject']
			if subjects:
				labels = []
				for label in subjects:
					labels.append(label['name'])
				label_list.append(labels)
			else:
				label_list.append('ZNONE')

	return article_list, label_list


# Create, compile, fit and save a LSTM model
def train_classifier(train_padded, trainy_seq, test_padded, testy_seq, label_amount, label_seq_to_label_dic):
	# Source used for help with implementation:
	# https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35

	# baseline
	dummy_clf = DummyClassifier(strategy="most_frequent")
	dummy_clf.fit(train_padded, trainy_seq)
	yguess = dummy_clf.predict(test_padded)
	print('Accuracy score majority class baseline: {0}'.format(accuracy_score(testy_seq, yguess)))

	print("Training Classifier")

	vocab_size = 5000
	embedding_dim = 64
	num_epochs = 5
	batch_size = 50

	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
		tf.keras.layers.Dense(embedding_dim, activation='selu'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(label_amount, activation='softmax')
	])

	model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	history = model.fit(train_padded, trainy_seq, epochs=num_epochs, batch_size=batch_size, validation_data=(test_padded, testy_seq), verbose=2)
	model.save("model/final_model")

	return model


# Pads all sequences to be at the same length to allow use with the LSTM model
def sequence_padding(trainx, testx, devx):
	vocab_size = 5000
	max_length = 200
	trunc_type = 'post'
	padding_type = 'post'
	oov_tok = '<OOV>'

	tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
	tokenizer.fit_on_texts(trainx)
	word_index = tokenizer.word_index
	dict(list(word_index.items())[0:10])

	train_sequences = tokenizer.texts_to_sequences(trainx)
	train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

	test_sequences = tokenizer.texts_to_sequences(testx)
	test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

	dev_sequences = tokenizer.texts_to_sequences(devx)
	dev_padded = pad_sequences(dev_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

	return train_padded, test_padded, dev_padded


# Load a previous classifier model at its default saving location
def load_classifier(path="model/final_model"):
	print("Loading Classifier")

	model = keras.models.load_model(path)

	return model


# Perform and print the evaluation of a given model
def classifier_evaluate(classifier, label_seq_to_label_dic, test_padded, testy_seq):
	# turn predictions to labels
	y_pred_per = classifier.predict(test_padded)
	y_pred = np.argmax(y_pred_per, axis=1)
	y_pred_labels = []
	for seq in y_pred:
		y_pred_labels.append(label_seq_to_label_dic[seq])

	y_test_labels = []
	for seq in testy_seq:
		y_test_labels.append(label_seq_to_label_dic[seq[0]])
	y_test_labels = np.asarray(y_test_labels)

	# print classification report
	print(classification_report(y_test_labels, y_pred_labels))
	print(confusion_matrix(y_test_labels, y_pred_labels))


# Turn labels into sequence to allow use with LSTM model
def labels_to_sequences(label_list, label_set):
	seq_index_counter = 0
	label_seq_to_label_dic = {}
	label_seq_dic = {}
	label_seq_list = []

	# create label to sequence dic
	for label in label_set:
		label_seq_dic[label] = seq_index_counter
		label_seq_to_label_dic[seq_index_counter] = label
		seq_index_counter += 1


	for labels in label_list:
		temp_list = []
		for label in labels:
			# use label sequence if in label set
			if label in label_set:
				# add sequence to article label list
				temp_list.append(label_seq_dic[label])

				# only add one label
				break

		# add none
		if len(temp_list) == 0:
			temp_list.append(label_seq_dic['ZNONE'])

		label_seq_list.append(temp_list)

	return label_seq_list, label_seq_to_label_dic


def most_common_labels(label_list, label_amount):
	label_count_dic = defaultdict(lambda: 0)
	for labels in label_list:
		for label in labels:
			label_count_dic[label] += 1

	sorted_list = sorted(label_count_dic, key=label_count_dic.get, reverse=True)

	return_set = set(sorted_list[0:label_amount])
	return_set.add('ZNONE')

	return return_set


def main(argv):
	# First argument is train file
	# Second argument is test file
	# Third time is whether to train a new model or load a previous model
	# When either train file or test file is set to None, the system will regard all files in the data folder
	# Prefered use is using no defined train or test file

	# Command line arguments
	train_file = None
	if len(argv) > 1:
		if argv[1].lower != "none":
			train_file = argv[1]
	test_file = None
	if len(argv) > 2:
		if argv[2].lower != "none":
			test_file = argv[2]
	train = True
	if len(argv) > 3:
		if argv[3].lower() == "false":
			train = False
		else:
			train = True
			
	# Use entire corpus
	if test_file is None or train_file is None:
		article_list, label_list = read_corpus('data')
		label_amount = 20
		label_set = most_common_labels(label_list, label_amount-1)
		label_seq_list, label_seq_to_label_dic = labels_to_sequences(label_list, label_set)

		# split in train and test
		train_percentage = 0.7
		test_percentage = 0.2
		article_len = len(article_list)
		train_split_index = round(article_len * train_percentage)
		test_split_index = round(article_len * (train_percentage + test_percentage))

		label_tokenizer = Tokenizer()
		label_tokenizer.fit_on_texts(label_list)

		trainx = article_list[:train_split_index]
		trainy = label_seq_list[:train_split_index]
		testx = article_list[train_split_index:test_split_index]
		testy = label_seq_list[train_split_index:test_split_index]
		devx = article_list[test_split_index:]
		devy = label_seq_list[test_split_index:]

		trainy_seq = np.array(trainy)
		testy_seq = np.array(testy)
		devy_seq = np.array(devy)
	# Use defined train and test file
	else:
		train_article_list, train_label_list = read_file(train_file)
		test_article_list, test_label_list = read_file(test_file)

		label_amount = 20
		label_set = most_common_labels(train_label_list + test_label_list, label_amount-1)
		label_seq_list, label_seq_to_label_dic = labels_to_sequences(train_label_list + test_label_list, label_set)
		split_index = len(train_label_list)

		trainx = train_article_list
		trainy = label_seq_list[:split_index]
		testx = test_article_list
		testy = label_seq_list[split_index:]
		devx = testx
		devy = testy

		trainy_seq = np.array(trainy)
		testy_seq = np.array(testy)
		devy_seq = np.array(devy)

	# padding
	train_padding, test_padding, dev_padding = sequence_padding(trainx, testx, devx)

	# classifier
	classifier = False
	if train:
		classifier = train_classifier(train_padding, trainy_seq, dev_padding, devy_seq, label_amount, label_seq_to_label_dic)
	else:
		try:
			classifier = load_classifier()
		except:
			print("No previous model found, use argument 'True' to train a model.")

	# evaluate
	if classifier is not False:
		classifier_evaluate(classifier, label_seq_to_label_dic, test_padding, testy_seq)


if __name__ == "__main__":
	main(sys.argv)