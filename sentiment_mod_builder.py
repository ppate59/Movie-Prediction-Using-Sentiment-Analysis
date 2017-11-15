import nltk
import numpy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import state_union,wordnet
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):# Created a class to count votes out of 8 algorithm and give confidence value
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes / len(votes))
        return conf

short_pos = open("positive.txt","r").read()     #.txt file defined our positive sentiments for this project
short_neg = open("negative.txt","r").read()     #.txt file defined our negative sentiments for this project

all_words = []
documents = []
allowed_word_types = ["C","DT","EX","FW","IN","J","LS","MD","N","PDT","POS","PR","R","TO","V","WDT","WH","WP$","WRB"]

for p in short_pos.decode('utf-8').split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.decode('utf-8').split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_documents = open("documents.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(documents, save_documents)
save_documents.close()

short_pos_words = word_tokenize(short_pos.decode('utf-8'))
short_neg_words = word_tokenize(short_neg.decode('utf-8'))

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_word_features = open("featuresets.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_word_features = open("word_features5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(word_features, save_word_features)
save_word_features.close()

random.shuffle(featuresets)
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print "Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100
classifier.show_most_informative_features(15)

save_classifier = open("originalnaivebayes5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print "MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100

save_classifier = open("MNB_classifier5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print "BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100

save_classifier = open("BernoulliNB_classifier5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print "LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100

save_classifier = open("LogisticRegression_classifier5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print "LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100

save_classifier = open("LinearSVC_classifier5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print "SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100

save_classifier = open("SGDC_classifier5k.pickle","wb")      #Create pickle file to use in sentiment_mod.py
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()
