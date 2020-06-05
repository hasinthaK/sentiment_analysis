import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import dump
from nltk.classify import ClassifierI
from statistics import mode


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
        return conf


    # for training ----------------------------------------
    # one liner
    # docs = [list(movie_reviews.words(fileid), category)
    #         for category in movie_reviews.categories()
    #         for fileid in movie_reviews.fileids(category)]
docs = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        docs.append((list(movie_reviews.words(fileid)), category))

random.shuffle(docs)
# ---------------------------------------------------------

# print(docs[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(10))
# print(all_words["product"])

# limit all_words
word_features = list(all_words.keys())[:4000]


def find_features(doc):
    # doc - list of words
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), category) for (rev, category) in docs]

# featureset - first 1000 -> neg, second 1000 -> pos
# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]
training_set = featuresets[100:]
testing_set = featuresets[:100]


# uncomment below line - Initial training
classifier = nltk.NaiveBayesClassifier.train(training_set)
# save classifier to a file
dump.dump(classifier, "naivebayes")
# load from saved classifier after initial training
# classifier = dump.loadDump("naivebayes")
print("naive Bayes accuracy: ", nltk.classify.accuracy(
    classifier, testing_set) * 100)
# classifier.show_most_informative_features(10)


# MultinomialNB, GaussianNB, BernoulliNB

# train & save
MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
dump.dump(MultinomialNB_classifier, "MNB")
# load trained & saved file
# MultinomialNB_classifier = dump.loadDump("MNB")
print("MultinomialNB_classifier accuracy: ", nltk.classify.accuracy(
    MultinomialNB_classifier, testing_set) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
dump.dump(BernoulliNB_classifier, "BNB")
# BernoulliNB_classifier = dump.loadDump("BNB")
print("BernoulliNB_classifier accuracy: ", nltk.classify.accuracy(
    BernoulliNB_classifier, testing_set) * 100)


# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# dump.dump(LogisticRegression_classifier, "LR")
# # LogisticRegression_classifier = dump.loadDump("LR")
# print("LogisticRegression_classifier accuracy: ", nltk.classify.accuracy(
#     LogisticRegression_classifier, testing_set) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
dump.dump(SGDClassifier_classifier, "SGD")
# SGDClassifier_classifier = dump.loadDump("SGD")
print("SGDClassifier_classifier accuracy: ", nltk.classify.accuracy(
    SGDClassifier_classifier, testing_set) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
dump.dump(SVC_classifier, "SVC")
# SVC_classifier = dump.loadDump("SVC")
print("SVC_classifier accuracy: ", nltk.classify.accuracy(
    SVC_classifier, testing_set) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
dump.dump(LinearSVC_classifier, "LinearSVC")
# LinearSVC_classifier = dump.loadDump("LinearSVC")
print("LinearSVC_classifier accuracy: ", nltk.classify.accuracy(
    LinearSVC_classifier, testing_set) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
dump.dump(NuSVC_classifier, "NuSVC")
# NuSVC_classifier = dump.loadDump("NuSVC")
print("NuSVC_classifier accuracy: ", nltk.classify.accuracy(
    NuSVC_classifier, testing_set) * 100)


voted_classifier = VoteClassifier(
    classifier,
    MultinomialNB_classifier,
    BernoulliNB_classifier,
    # LogisticRegression_classifier,
    SVC_classifier,
    SGDClassifier_classifier,
    LinearSVC_classifier,
    NuSVC_classifier)

print("Voted_classifier accuracy: ", nltk.classify.accuracy(
    voted_classifier, testing_set) * 100)

# print("Classification: ", voted_classifier.classify(
#     testing_set[0][0]), "Confidence: ", voted_classifier.confidence(testing_set[0][0]) * 100)
