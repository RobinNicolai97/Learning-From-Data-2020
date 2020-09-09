from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys

# Checks the command line arguments for input
# sentiment - enables sentiment
# probability - shows predictions as probability
arg_sentiment = sys.argv[1].lower() == "sentiment"
arg_probability = sys.argv[2].lower() == "probability"

# COMMENT THIS
# This function takes the corpus textfile as an input, together with a Boolean expression (True/False).
# If use_sentiment == True, the script will classify reviews as positive or negative. If use_sentiment == False, it will classify reviews into six possible categories. 
# It takes the label (the to be predicted class) out of each line in the tokenized textfile ( tokens[1] for sentiment, tokens[0] for category ) 
# together with the textual content of the review: token[3:].
# The function returns a list of textual reviews and a list of their corresponding labels.


def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# COMMENT THIS
# The function read_corpus is activated, so that X == the textual content of the reviews and Y == their corresponding labels.
# The data is divided into 75% training-data and 25% test-data. 

X, Y = read_corpus('trainset.txt', use_sentiment=arg_sentiment)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier.
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# COMMENT THIS
# The Naive Bayes classifier takes the textual content of the reviews and their corresponding classes. 
# It will train on right combinations of review and classes. Then it will be able to make predictions about the classes of not-seen-before review texts. 
classifier.fit(Xtrain, Ytrain)

# COMMENT THIS  
# The Naive Bayes classifier predicts the classes of all reviews in the test-data.
Yguess = classifier.predict(Xtest)

# The script prints all probabilities for each feature set
if arg_probability:
    prob_list = classifier.predict_proba(Xtest)
    for prob_array in prob_list:
        print(prob_array)

# COMMENT THIS
# a simple accuracy measure is taken. This score is basicly how many times Ytest[X] == Yguess[X], divided by the length of Ytest and Yguess (which are both the same length).
print('\n accuracy score:', accuracy_score(Ytest, Yguess) ) 
print('\n\n') 
print(classification_report(Ytest,Yguess))
print('\n Confusion Matrix \n') 
print(confusion_matrix(Ytest,Yguess))
