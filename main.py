from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import numpy as np
from collections import Counter
import string
import re
import math

# Loading the file
with open("DevilsDictionary.txt", "r") as file:
    book = file.read().lower()

    # Remove everything except a-z and space
    book = re.sub('[^a-z ]+', '', book)

    # Cleaning the text from stopwords (but keep the spaces)
    stop_words = set(stopwords.words('english'))
    for stopword in stop_words:
        book = book.replace(' '+stopword+' ', ' ')
        
    # Finding letter frequency
    alphabet = list(string.ascii_lowercase)+[' ']

    letters_counter = Counter(book)

    print(letters_counter)
    
    total_chars = len(book)

    letters_freq = {}
    for k,v in letters_counter.items():
        letters_freq[k] = v*1.0 / total_chars

    print(letters_freq)

    # Calculating entropy
    entropy = 0
    for k,v in letters_freq.items():
        entropy += v*math.log(v, 2)

    entropy *= (-1)

    print(entropy)

    # Calculating 2 letters sequnces
    two_letters_counter = Counter()

    for i in xrange(len(book)-1):
        first_letter = book[i]
        second_letter = book[i+1]

        two_letters_counter[(first_letter, second_letter)] += 1

    # Number of tokens
    tokens = [w for w in book.split(' ') if w]

    print(len(tokens))

    # Number of different word (word types)
    diff_words = set(tokens)

    print(len(diff_words))

    # Word frequency
    word_freq = Counter(tokens)

