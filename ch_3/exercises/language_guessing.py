from nltk.corpus import udhr, gutenberg
from nltk import FreqDist, spearman_correlation, ranks_from_sequence
from sys import stdout

__author__ = 'mjholler'


ENCODINGS = ['latin-1', 'utf-8']
# ENCODINGS = ['latin-1']


def udhr_rankings(debug=False):
    """ Get the conditional frequency distributions for each language in the udhr corpus.
    :returns: dictionary of language to conditional frequency distribution
    :rtype: dict
    """
    result = dict()

    if debug:
        stdout.write('Preparing training sets')

    for _id in [s for s in udhr.fileids() if '-' in s]:
        split_id = _id.split('-')
        language = split_id[0]

        # Only allow some encodings.
        if udhr.encoding(_id) not in ENCODINGS:
            continue

        try:
            words = udhr.words(_id)
            result[language] = FreqDist(words)
        except AssertionError:
            # Problems reading, so we skip.
            pass
        except UnicodeDecodeError:
            # Problems reading, so we skip.
            pass

        if debug:
            stdout.write('.')
            stdout.flush()

    if debug:
        stdout.write('\n')

    return result


def predict_language(sample_fd, training_set_fds, debug=False):
    """ Predict language using the spearman coefficient trained on UDHR translations.
    :param sample_fd:
    :type sample_fd: FreqDist
    :param training_set_fds: Dictionary of language to frequency distribution.
    :type training_set_fds: dict
    :returns: Best matching language.
    :rtype: str
    """
    scores = dict()

    if debug:
        stdout.write('Finding best match')

    for language, language_fd in training_set_fds.iteritems():
        # make copies so we don't alter the originals
        sfd = dict(sample_fd)
        lfd = dict(language_fd)

        # make sure both frequency distributions have only the keys they have in common
        # delete_differences(sfd, lfd)

        scores[language] = spearman_correlation(
            ranks_from_sequence(sfd),
            ranks_from_sequence(lfd)
        )

        if debug:
            stdout.write('.')
            stdout.flush()

    stdout.write('\n')

    return sorted(scores.items(), key=lambda x: x[-1], reverse=True)


def delete_differences(d1, d2):
    """
    :param d1:
    :type d1: dict
    :param d2:
    :type d2: dict
    """
    same = set(d1.keys()) & set(d2.keys())

    for k in set(d1.keys()) - same:
        del d1[k]

    for k in set(d2.keys()) - same:
        del d2[k]


def main():
    sample_rankings = FreqDist(gutenberg.words('austen-persuasion.txt'))
    training_set_rankings = udhr_rankings(debug=True)

    predictions = predict_language(sample_rankings, training_set_rankings, debug=True)

    print
    for language, value in predictions:
        if value != 0:
            # print '%.-32s\t%.-10s' % (language, value)
            print '{:.<32}{}'.format(language, value)


if __name__ == '__main__':
    main()


