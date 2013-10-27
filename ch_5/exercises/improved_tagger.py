import nltk
from nltk.corpus import brown

__author__ = 'mjholler'


class PreviousUnigramTagger(nltk.UnigramTagger):

    def context(self, tokens, index, history):
        """
        In this tagger, context is only relative to the previous token,
        not the current one. Thus, we return the previous token if there
        is one, or else we let it fall to the backoff tagger by returning
        None.
        """
        return tokens[index - 1] if index else None


def original_tagger(train_sets):
    default = nltk.DefaultTagger('NN')
    unigram = nltk.UnigramTagger(train_sets, backoff=default)
    bigram = nltk.BigramTagger(train_sets, backoff=unigram)
    return nltk.TrigramTagger(train_sets, backoff=bigram)


def modified_tagger(train_sets):
    default = nltk.DefaultTagger('NN')
    previous = PreviousUnigramTagger(train_sets, backoff=default)
    unigram = nltk.UnigramTagger(train_sets, backoff=previous)
    bigram = nltk.BigramTagger(train_sets, backoff=unigram)
    return nltk.TrigramTagger(train_sets, backoff=bigram)


def main():
    # Establish the texts we'll be using
    brown_tagged_sents = brown.tagged_sents(categories='news')

    # Set up training and test sets
    size = int(len(brown_tagged_sents) * 0.9)
    train_sets = brown_tagged_sents[:size]
    test_sets = brown_tagged_sents[size:]

    print 'Original:', original_tagger(train_sets).evaluate(test_sets)
    print 'Modified:', modified_tagger(train_sets).evaluate(test_sets)


if __name__ == '__main__':
    main()
