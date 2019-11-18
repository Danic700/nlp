from nltk.corpus import stopwords
import numpy as np
from collections import Counter

# Loading the file
file = open("DevilsDictionary.txt", "r")
book = file.read().lower()

# Cleaning the text
stop_words = stopwords.words('english')
clean_word_list = [word for word in book.split() if word not in stop_words]

# Finding letter frequency
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', '']

letters = Counter(''.join(clean_word_list))
filtered_items = [(k, v) for (k, v) in letters.items() if k in alphabet]
filtered_items = dict(filtered_items)
num_of_characters = len(clean_word_list.__str__())
for k, v in filtered_items.items():
    filtered_items[k] = v/num_of_characters


# Calculating entropy





print(filtered_items)
print(clean_word_list)
print("Num of charcters", num_of_characters)