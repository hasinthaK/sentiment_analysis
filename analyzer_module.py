import nltk
import random
import string
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import dump
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
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
        conf = choice_votes / len(votes)
        return format(conf, ".2%")


# ---------------------------------------------------------
word_features = dump.loadDump("word_features")


def find_features(doc):
    # doc - list of words
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


classifier = dump.loadDump("naivebayes")
MultinomialNB_classifier = dump.loadDump("MNB")
BernoulliNB_classifier = dump.loadDump("BNB")
SGDClassifier_classifier = dump.loadDump("SGD")
SVC_classifier = dump.loadDump("SVC")
LinearSVC_classifier = dump.loadDump("LinearSVC")
NuSVC_classifier = dump.loadDump("NuSVC")


voted_classifier = VoteClassifier(
    classifier,
    MultinomialNB_classifier,
    BernoulliNB_classifier,
    # LogisticRegression_classifier,
    SVC_classifier,
    SGDClassifier_classifier,
    LinearSVC_classifier,
    NuSVC_classifier)


def sentiment(review):
    cleaned_review = review.translate(
        str.maketrans("", "", string.punctuation))
    feat = find_features(word_tokenize(cleaned_review))
    print(voted_classifier.classify(feat),
          voted_classifier.confidence(feat))


text = "Very nice one, Came very quickly, I recommend this seller & also the product, but the service was not the best!"
# text = "only in its final surprising shots does rabbit-proof fence find the authority it's looking for . "
sentiment(text)
