from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import time
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import ngrams, pos_tag
from sklearn.svm import SVC, LinearSVC

# Checks the command line arguments for input


if len(sys.argv) < 3:
    print('Please use format: LFDassignment2.py traindata testdata') 
    exit()
    
trainset = sys.argv[1] 
testset = sys.argv[2]


# This function takes the corpus textfile as an input
# It takes the label (the to be predicted class) out of each line in the tokenized textfile ( tokens[1] for sentiment, tokens[0] for category ) 
# together with the textual content of the review: token[3:] (which is being processed further into tokens_processed.
# The function returns a list of textual reviews and a list of their corresponding labels.


def read_corpus(corpus_file):
    documents = []
    labels = []
    n = 2
    #stop_words = set(stopwords.words('english')) 
    #stemmer = PorterStemmer()
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens_processed = []
            tokens = line.strip().split()
            token_review = tokens[3:]
            #for word in token_review: 
                #if word not in stop_words and word.isalpha(): #removing stopwords and non-alphabetic string. 
                    #word = stemmer.stem(word) #stemming words
                    #tokens_processed.append(word)
            bigrams = ngrams(tokens[3:], n)
                    
            documents.append(pos_tag(tokens[3:]))
            # 2-class problem: positive vs negative
            labels.append( tokens[1] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x


# The function read_corpus is activated, so that X == the textual content of the reviews and Y == their corresponding labels.

Xtrain, Ytrain = read_corpus(trainset)
Xtest, Ytest = read_corpus(testset)

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
                        ('cls', LinearSVC(C = 0.8))] )
                        
# 
# The Naive Bayes classifier takes the textual content of the reviews and their corresponding classes. 


classifier.fit(Xtrain, Ytrain)

# The Naive Bayes classifier predicts the classes of all reviews in the test-data.

Yguess = classifier.predict(Xtest)




    # a simple accuracy measure is taken. This score is basicly how many times Ytest[X] == Yguess[X], divided by the length of Ytest and Yguess (which are both the same length).
print('\n accuracy score:', accuracy_score(Ytest, Yguess) )
print('\n\n')
print(classification_report(Ytest,Yguess))
print('\n Confusion Matrix \n')
print(confusion_matrix(Ytest,Yguess))
