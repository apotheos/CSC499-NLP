import random

import nltk
from nltk.corpus import movie_reviews, wordnet
from nltk.classify import apply_features

__author__ = 'mjholler'


def synsets(words):
    syns = set()
    for w in words:
        syns.update(str(s) for s in wordnet.synsets(w))

    return syns

# retrieve all movie reviews in the form of (wordlist, category)
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# randomize the documents so any default category ordering is removed
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:2000]

# Get the top synsets in the document from the top 2000 words
synset_features = synsets(word_features)


def document_features(document):
    document_words = set(document)
    document_synsets = synsets(document_words)

    for word in document_words:
        document_synsets.update(str(s) for s in wordnet.synsets(word))

    features = dict()

    # for word in word_features:
    #     features['contains({})'.format(word)] = (word in document_words)

    for synset in synset_features:
        features[synset] = (synset in document_synsets)

    return features

train_set, test_set = apply_features(document_features, documents[100:]), apply_features(document_features, documents[:100])

print 'training classifier'
classifier = nltk.NaiveBayesClassifier.train(train_set)

print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(10)
