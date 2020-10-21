import json
import os
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


def read_corpus(dir_name='data'):
	file_list = os.listdir(dir_name)
	article_list = []
	label_list = []

	for file in file_list:
		with open(dir_name + "\\" + file) as f:
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


def train_classifier(trainx, trainy_seq, testx, testy_seq, label_amount):
	vocab_size = 5000
	embedding_dim = 64
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

	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
		tf.keras.layers.Dense(embedding_dim, activation='relu'),
		tf.keras.layers.Dense(label_amount, activation='softmax')
	])

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	num_epochs = 10
	history = model.fit(train_padded, trainy_seq, epochs=num_epochs, validation_data=(test_padded, testy_seq), verbose=2)


def labels_to_sequences(label_list, label_set):
	seq_index_counter = 0
	label_seq_dic = {}
	label_seq_list = []

	# create label to sequence dic
	for label in label_set:
		label_seq_dic[label] = seq_index_counter
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

	return label_seq_list, label_seq_dic


def most_common_labels(label_list, label_amount):
	label_count_dic = defaultdict(lambda: 0)
	for labels in label_list:
		for label in labels:
			label_count_dic[label] += 1

	sorted_list = sorted(label_count_dic, key=label_count_dic.get, reverse=True)

	return_set = set(sorted_list[0:label_amount])
	return_set.add('ZNONE')

	return return_set


def main():
	article_list, label_list = read_corpus('data')
	label_amount = 20
	label_set = most_common_labels(label_list, label_amount-1)
	label_seq_list, label_seq_dic = labels_to_sequences(label_list, label_set)

	# split in train and test
	train_percentage = 0.8
	article_len = len(article_list)
	split_index = round(article_len * train_percentage)

	label_tokenizer = Tokenizer()
	label_tokenizer.fit_on_texts(label_list)

	trainx = article_list[:split_index]
	trainy = label_seq_list[:split_index]
	testx = article_list[split_index:]
	testy = label_seq_list[split_index:]

	trainy_seq = np.array(trainy)
	testy_seq = np.array(testy)

	for item in testy_seq:
		print(item)

	# classifier
	classifier = train_classifier(trainx, trainy_seq, testx, testy_seq, label_amount)


if __name__ == "__main__":
	main()
