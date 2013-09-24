#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import nltk
from nltk.corpus import PlaintextCorpusReader

CORPUS_ROOT_POKEMON = '../../corpora/pokemon'
CORPUS_ROOT_SHERLOCK = '../../corpora/sherlock-holmes'


def generate_model(cfdist, word, num=15):
    for i in xrange(num):
        print word,
        # word = cfdist[word].max()
        word = random.choice(list(cfdist[word])[:5])


def sherlock_cfdist():
    sherlock_corpus = PlaintextCorpusReader(CORPUS_ROOT_SHERLOCK, '.*', encoding='utf-8')
    sherlock_bigrams = nltk.bigrams(sherlock_corpus.words())

    return nltk.ConditionalFreqDist(sherlock_bigrams)


def pokemon_cfdist():
    pokemon_corpus = PlaintextCorpusReader(CORPUS_ROOT_POKEMON, '.*', encoding='utf-8')
    pokemon_bigrams = nltk.bigrams(pokemon_corpus.words())

    return nltk.ConditionalFreqDist(pokemon_bigrams)


def hybrid_cfdist():
    sherlock_corpus = PlaintextCorpusReader(CORPUS_ROOT_SHERLOCK, '.*', encoding='utf-8')
    sherlock_bigrams = nltk.bigrams(sherlock_corpus.words())

    pokemon_corpus = PlaintextCorpusReader(CORPUS_ROOT_POKEMON, '.*', encoding='utf-8')
    pokemon_bigrams = nltk.bigrams(pokemon_corpus.words())

    return nltk.ConditionalFreqDist(sherlock_bigrams + pokemon_bigrams)




def main():
    print 'Problem 24B:'
    generate_model(sherlock_cfdist(), 'elementary', num=50)
    print ''
    generate_model(sherlock_cfdist(), 'gun', num=50)

    print '\n\nProblem 24C:'
    generate_model(hybrid_cfdist(), 'Pikachu', num=50)
    print ''
    generate_model(hybrid_cfdist(), 'Squirtle', num=50)
    print ''
    generate_model(hybrid_cfdist(), 'Sherlock', num=50)



if __name__ == '__main__':
    main()

