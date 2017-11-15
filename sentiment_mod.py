import nltk,numpy,random,pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import state_union,wordnet
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI): # Created a class to count votes out of 8 algorithm and give confidence value
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
        conf = float(choice_votes) / float(len(votes))
        return format(conf,'.2f')

documents_f = open("documents.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("word_features5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets_f = open("featuresets.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

open_file = open("originalnaivebayes5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
classifier = pickle.load(open_file)
open_file.close()

open_file = open("MNB_classifier5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("BernoulliNB_classifier5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LogisticRegression_classifier5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LinearSVC_classifier5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("SGDC_classifier5k.pickle", "rb")    #Use pickle file created in sentiment_mod_builder.py
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier1,
                                  SGDC_classifier)

def sentiment(text):    #This function will used in tweet sentiment analysis and return sentiment and confidence value
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
