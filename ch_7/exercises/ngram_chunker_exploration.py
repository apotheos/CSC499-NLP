import re
from time import sleep
import nltk
from nltk.corpus import conll2000

__author__ = 'mjholler'


def np_tags_fd(sents):

    fd = nltk.FreqDist()

    for sent in sents:
        current_np = list()
        for w, t, c in sent:
            if c == 'O':
                if current_np:
                    fd.inc(tuple(current_np))
                    current_np = list()
            else:
                current_np.append(t)

    return fd


def print_frequencies(fd, num_results=None):
    """
    @param fd: freq dist
    @type fd: FreqDist
    """

    fmt = "{0:.4f}% \t{1}"
    n = fd.N()

    num_results = num_results if num_results else fd.N()

    for pattern in sorted(fd.keys()[:num_results]):
        rel_freq = fd[pattern] / float(n) * 100
        print fmt.format(rel_freq, pattern)


def regex_generator(fd):
    """
    @param fd: freq dist
    @type fd: FreqDist
    """

    regex = ['NP: ']

    for pattern in sorted(fd.keys()[:int(fd.B() * 0.20)]):
        regex.append('{')
        for pos in pattern:
            if pos not in [',', ':']:
                regex.append('<')
                regex.append(re.escape(pos))
                regex.append('>')

        regex.append('}\n    ')

    return ''.join(regex)


def main():
    train_sents = (nltk.chunk.tree2conlltags(s) for s in conll2000.chunked_sents('train.txt', chunk_types=['NP']))
    # test_sents = (nltk.chunk.tree2conlltags(s) for s in conll2000.chunked_sents('test.txt', chunk_types=['NP']))
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

    fd = np_tags_fd(train_sents)
    print_frequencies(fd, num_results=50)
    # pattern = regex_generator(fd)
    # print pattern
    # pattern = r"NP: {<NN>}"

    print nltk.RegexpParser("").evaluate(test_sents)
    print ''

    pattern_book = r"NP: {<[CDJNP].*>+}"
    print nltk.RegexpParser(pattern_book).evaluate(test_sents)
    print ''

    pattern_modified = r"NP: {<(\$)>?<[CDJNP].*>+}"
    print nltk.RegexpParser(pattern_modified).evaluate(test_sents)
    print ''

    pattern_modified = r"""NP: {<(\$)>?<[CDJNP].*>+}
                               {<W(P|DT)>}"""
    print nltk.RegexpParser(pattern_modified).evaluate(test_sents)

    # cp = nltk.RegexpParser(pattern)

    # print cp.evaluate(test_sents)

    # print NGramChunker(train_sets, nltk.UnigramTagger).evaluate(test_sets)
    # print NGramChunker(train_sets, nltk.BigramTagger).evaluate(test_sets)
    # print NGramChunker(train_sets, nltk.TrigramTagger).evaluate(test_sets)


if __name__ == '__main__':
    main()