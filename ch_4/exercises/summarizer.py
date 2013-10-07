import copy
from nltk import FreqDist
from nltk.corpus import brown, stopwords, reuters, gutenberg, PlaintextCorpusReader
from nltk.stem.snowball import EnglishStemmer


def remove_stopwords(words):
    """
    :type words: Sequence
    :rtype: Sequence
    """
    return [w for w in words if not w in stopwords.words('english')]


def stem(words):
    """
    :type words: Sequence
    :rtype: Sequence
    """
    stemmer = EnglishStemmer()
    return [stemmer.stem(w) for w in words]


def preprocessor(sents):
    """ Proprocess text by running it through stopword removal and a stemmer.
    :type sents: list
    """
    for i, sent in enumerate(sents):
        sents[i] = stem(remove_stopwords(sent))


def groups_to_words(groups):
    """ Flatten a grouping to a list of words. Generator function.
    :param groups: a list of groupings, where each grouping contains words
    :type groups: list
    :rtype: Sequence
    """
    for group in groups:
        for word in group:
            yield word


class ScoringStrategy(object):
    def execute(self, sent, fd):
        pass


class TotalScoreStrategy(object):
    """ Calculates score by summing frequencies in sentence.
    """
    def execute(self, sent, fd):
        return sum(fd[word] for word in sent)


class DensityScoreStrategy(object):
    """ Calculates score by summing frequencies in sentence, then dividing by wordcount.
    """
    def execute(self, sent, fd):
        return sum(fd[word] for word in sent) / len(sent)


class Summary(object):

    def __init__(self, sents, preprocessor=None):
        self.sents = sents
        self.preprocessed_sents = copy.deepcopy(list(sents))

        if preprocessor:
            preprocessor(self.preprocessed_sents)

    def summarize(self, strategy, n=3):
        fd = FreqDist(groups_to_words(self.preprocessed_sents))

        # Find the top n sentences with the highest total summary score, ordered by score
        scored_sents = (ScoredSentence(s, i, fd, strategy) for i, s in enumerate(self.preprocessed_sents))
        top_sents = sorted(scored_sents, reverse=True)[:n]

        # Find where the order the top sentences by appearance
        return sorted(((self.sents[s.position], s.score) for s in top_sents))


class ScoredSentence(object):

    def __init__(self, sent, position, fd, strategy):
        """ Score the sentence against a given frequency distribution.
        :param sent: list of words making up a sentence
        :type sent: list
        :param position: integer position of sentence in article
        :type position: int
        :param fd: frequency distribution to score the sentence against
        :type fd: FreqDist
        :param strategy: Strategy used to calculate scores
        :type strategy: ScoringStrategy
        """
        self.sent = sent
        self.position = position
        self.score = strategy.execute(sent, fd)

    def __cmp__(self, other):
        """
        :type other: ScoredSentence
        """
        return self.score - other.score


def main():
    corpus = PlaintextCorpusReader('../../corpora/news', '.*', encoding='UTF-8')

    article_sents = corpus.sents(fileids='gravity_review.txt')
    # article_sents = corpus.sents(fileids='wwII.txt')

    summary = Summary(article_sents, preprocessor)
    # summary = Summary(article_sents)

    print '<<<<<<<<<<<<<<<<<<<<< DENSITY SCORING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    strategy = DensityScoreStrategy()
    for sent, score in summary.summarize(strategy, n=3):
        print score, ' '.join(sent), '\n'

    print '<<<<<<<<<<<<<<<<<<<<< TOTAL SCORING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    strategy = TotalScoreStrategy()
    for sent, score in summary.summarize(strategy, n=3):
        print score, ' '.join(sent), '\n'


if __name__ == '__main__':
    main()