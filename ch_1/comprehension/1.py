__author__ = 'mjholler'

from nltk.book import *

all_tokens = (text1.tokens + text2.tokens + text3.tokens + text4.tokens + text5.tokens + text6.tokens +
              text7.tokens + text8.tokens + text9.tokens)

all_dist = FreqDist([len(x) for x in all_tokens]).plot()

