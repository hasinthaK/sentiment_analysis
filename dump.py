import pickle


def dump(classifier, classifier_name):
    save_classifier = open("trained/" + classifier_name + ".pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


def loadDump(classifier_name):
    classifier_f = open("trained/" + classifier_name + ".pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier
