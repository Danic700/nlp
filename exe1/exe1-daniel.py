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


file = open("DevilsDictionary.txt", "r")
book = file.read().lower()

# Remove everything except a-z and space
book = re.sub('[^a-z ]+', ' ', book)

# Cleaning the text from stopwords (but keep the spaces)
stop_words = set(stopwords.words('english'))
for stopword in stop_words:
    book = book.replace(' ' + stopword + ' ', ' ')

# (4a) Finding letter frequency
letters_counter = Counter(book)
total_chars = len(book)
letters_freq = {}
for k, v in letters_counter.items():
    letters_freq[k] = v * 1.0 / total_chars

# Calculating entropy
entrop = entropy(list(letters_freq.values()), base=2)
print("This is the entropy for the distribution of a letter in the text", entrop)

# Calculating cross-entropy
letter_distribution = []
for i in range(0, 27):
    letter_distribution.append(1 / 27)

cross_entropy = entropy(list(letters_freq.values()), letter_distribution, base=2) + entrop
print("This is the cross entropy for the distribution for a letter in the text and a letter in general", cross_entropy)

# (4b) Calculating 2 letters sequnces
two_letters_counter = Counter()

for i in xrange(len(book) - 1):
    first_letter = book[i]
    second_letter = book[i + 1]
    two_letters_counter[(first_letter, second_letter)] += 1

for k, v in two_letters_counter:
    two_letters_counter[(k, v)] /= letters_counter[k]

entrop = entropy(list(two_letters_counter.values()), base=2)
print("This is the entropy for the distribution based on the prior letter appearance", entrop)

two_letter_distribution = []
for i in range(0, 538):
    two_letter_distribution.append(1 / 729)

cross_entropy = entropy(list(two_letters_counter.values()), two_letter_distribution, base=2) + entrop
print("This is the cross entropy of the distribution based on the prior letter and two letters in general ",
      cross_entropy)

# (5) Number of tokens
tokens = word_tokenize(book)
print(len(tokens))

# Number of different word (word types)
diff_words = set(tokens)
print(len(diff_words))

# Word frequency
word_freq = Counter(tokens)
word_freq = Counter({k: v for k, v in word_freq.items() if len(k) > 1})  # Further Cleaning