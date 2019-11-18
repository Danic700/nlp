from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter


# Loading the file
file = open("DevilsDictionary.txt", "r")
book = file.read().lower()

# Cleaning the text
stop_words = stopwords.words('english')
clean_word_list = [word for word in book.split() if word not in stop_words]

# Finding letter frequency
letters = Counter(''.join(clean_word_list))

print(letters)
print(clean_word_list)

