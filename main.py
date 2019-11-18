from nltk.corpus import stopwords # Required for cleaning the text
from scipy  import stats # Requried for calculating entropy


def letterFrequency(txt):
    from collections import Counter
    from string import ascii_lowercase
    c=Counter(txt.lower())
    n=len(txt)/100.
    ascii_lowercase += "."
    print(ascii_lowercase)
    return [(x, c[x]/n) for x in ascii_lowercase]

# Loading the file
file = open("DevilsDictionary.txt", "r")
book = file.read()

# Cleaning the text
stop_words = stopwords.words('english')
clean_word_list = [word for word in book.split() if word not in stop_words]

# Calculate a-z & '.' letters frequency
lf = letterFrequency(' '.join(map(str,clean_word_list)))
print(lf)

# Get 2 lists: letters and their frequency
qk, pk = zip(*lf)

# Estimate the entropy of english letters based on the current corpus
englishEntopyEstimate = stats.entropy(pk)
print(englishEntopyEstimate)

