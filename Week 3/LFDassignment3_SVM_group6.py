from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
import sys

# NLTK imports
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Input arguments
if len(sys.argv) < 3:
    print('Please use format: LFDassignment3_SVM_group6.py <traindata> <testdata>')
    exit()

arg_train_file = sys.argv[1]
arg_test_file = sys.argv[2]


# Function that does basic feature selection
def feature_selection(feature_sets, lemmatizer, stemmer):
    new_feature_sets = []
    for feature_set in feature_sets:
        # Perform lemmatization
        features_lemmatized = []
        for word in feature_set:
            features_lemmatized.append(lemmatizer.lemmatize(word))

        # Perform stemming
        features_stemmed = []
        for word in features_lemmatized:
            features_stemmed.append(stemmer.stem(word))

        new_features = features_stemmed

        new_feature_sets.append(new_features)

    return new_feature_sets


# Read corpus function
def read_corpus(corpus_file, use_sentiment=True):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

    return documents, labels


# a dummy function that just returns its input
def identity(x):
    return x


# The function read_corpus is activated, so that X == the textual content of the reviews and Y == their corresponding labels.
Xtrain, Ytrain = read_corpus(arg_train_file)
Xtest, Ytest = read_corpus(arg_test_file)

# Init resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Perform basic feature selection on X sets
Xtrain = feature_selection(Xtrain, lemmatizer, stemmer)
Xtest = feature_selection(Xtest, lemmatizer, stemmer)

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor=identity,
                          tokenizer=identity)
else:
    vec = CountVectorizer(preprocessor=identity,
                          tokenizer=identity)

# combine the vectorizer with a Naive Bayes classifier.
classifier = Pipeline( [('vec', vec),
                        ('cls', SVC(C=3, gamma=0.8))])

# The Naive Bayes classifier takes the textual content of the reviews and their corresponding classes.
classifier.fit(Xtrain, Ytrain)

# The Naive Bayes classifier predicts the classes of all reviews in the test-data.
Yguess = classifier.predict(Xtest)

# A simple accuracy measure is taken. This score is basicaly how many times Ytest[X] == Yguess[X], divided by the length of Ytest and Yguess (which are both the same length).
print('\n accuracy score:', accuracy_score(Ytest, Yguess))
print('\n\n')
print(classification_report(Ytest, Yguess))
print('\n Confusion Matrix \n')
print(confusion_matrix(Ytest, Yguess))
