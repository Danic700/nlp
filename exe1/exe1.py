# Internal imports
import exe1_q4, exe1_q5

# External imports
from string import ascii_lowercase
from nltk.corpus import stopwords  # Required for cleaning the text

# Loading the file
file = open("DevilsDictionary.txt", "r")
book = file.read()

# Get stop words.
stop_words = set(stopwords.words("english")).union(('n.', 'adj'))

# Define valid letters: a-z and backspace
valid_letters = ascii_lowercase + " "

exe1_q4.question4(book, valid_letters, stop_words)
exe1_q5.question5(book, valid_letters, stop_words)
