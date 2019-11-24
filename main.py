from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from scipy._lib.six import xrange
from scipy.stats import entropy
import string
import re
import math

# Loading the file


with open("DevilsDictionary.txt", "r") as file:
    book = file.read().lower()

    # Remove everything except a-z and space
    book = re.sub('[^a-z ]+', ' ', book)

    # Cleaning the text from stopwords (but keep the spaces)
    stop_words = set(stopwords.words('english'))
    for stopword in stop_words:
        book = book.replace(' ' + stopword + ' ', ' ')

    # (4a) Finding letter frequency
    letters_counter = Counter(book)
    print(letters_counter)
    total_chars = len(book)
    letters_freq = {}
    for k, v in letters_counter.items():
        letters_freq[k] = v * 1.0 / total_chars

    print(letters_freq)

    # Calculating entropy
    entrop = entropy(list(letters_freq.values()), base=2)
    print(entrop)

    # (4b) Calculating 2 letters sequnces
    two_letters_counter = Counter()

    for i in xrange(len(book) - 1):
        first_letter = book[i]
        second_letter = book[i + 1]

        two_letters_counter[(first_letter, second_letter)] += 1

    for k, v in two_letters_counter:
        two_letters_counter[(k, v)] /= letters_counter[k]

    entrop = entropy(list(two_letters_counter.values()), base=2)
    print(entrop)

    # (5) Number of tokens
    tokens = word_tokenize(book)
    print(len(tokens))
    print(tokens)

    # Number of different word (word types)
    diff_words = set(tokens)
    print(len(diff_words))

    # Word frequency
    word_freq = Counter(tokens)
    print(word_freq.most_common())
