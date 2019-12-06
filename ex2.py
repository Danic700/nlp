from nltk.corpus import stopwords
from collections import Counter, defaultdict
import string
import re
import math
import os

######################################## Helper functions ########################################

def get_words(text):
      return text.split()

def clean_text(text, removeDot):
      r = '[^a-z ]+'
      if removeDot:
            '[^a-z \.]+'
      # leave only english chars
      text = re.sub(r, ' ', text)

      # Cleaning the text from stopwords (but keep the spaces)
      stop_words = set(stopwords.words('english'))
      for stopword in stop_words:
            text = text.replace(' ' + stopword + ' ', ' ')

      return text

def read_text(dirname):
      text = ''
      for filename in os.listdir(dirname):
            file = open(dirname+'/'+filename, "r")
            text += file.read().lower()+'\n'

      return text

def read_dictionary():
      return open('word_list_20k.txt', 'r').read().split('\n')


######################################## Unigram model ########################################

print("######################################## Unigram model ########################################")

# Train
print("Reading train texts")
text = read_text('train')
print("Cleaning train texts")
text = clean_text(text, True)
print("Reading dictionary")
dictionary = read_dictionary()
counters = Counter()
total_count = 0
words = get_words(text)
print("Counting words "+str(len(words)))
for word in words:
      total_count += 1
      if word in dictionary:
            counters[word] += 1
      else:
            counters['UNK'] += 1
word_freq = {}
print("Calculating word frequency")
for word in counters.keys():
      word_freq[word] = counters[word] / (total_count * 1.0)

# Test
print("Reading test text")
text = read_text('test')
print("Cleaning test text")
text = clean_text(text, False)
total_words = 0
sum = 0
words = get_words(text)
print("Counting words "+str(len(words)))
for word in words:
      total_words += 1
      if word in word_freq.keys():
            sum += -math.log(word_freq[word],2)
print("Entropy="+str((sum / total_words)))


######################################## Bigram model ########################################

print("######################################## Bigram model ########################################")

# Train
print("Reading train texts")
text = read_text('train')
print("Cleaning train texts")
text = clean_text(text, False)
print("Reading dictionary")
dictionary = read_dictionary()
counters = Counter()
total_count = 0
sentences = text.split('.')
print("Counting sentences "+str(len(sentences)))
counts = Counter()
context_counts = Counter()
for sentence in sentences:
      words = get_words(sentence)
      for i in xrange(1, len(words)):
            # bigram count and count context
            counts[(words[i-1], words[i])] += 1
            context_counts[(words[i-1])] += 1

            # unigram count and count context
            counts[(words[i])] += 1
            context_counts[()] += 1

print("Calculating probabilities")
probabilities = defaultdict(lambda: 0)
for ngram in counts.keys():
      count = counts[ngram]
      context = list(ngram)
      context.pop()
      context = tuple(context)
      probability = (count * 1.0) / context_counts[context]
      probabilities[ngram] = probability

# Test
print("Reading test text")
text = read_text('test')
print("Cleaning test text")
text = clean_text(text, False)
sentences = text.split('.')
# Linear interpulation
lambda_1 = 0.2
lambda_2 = 0.8
H = 0
W = 0
print("Counting sentences "+str(len(sentences)))
for sentence in sentences:
      words = get_words(sentence)
      for i in xrange(1, len(words)):
            P1 = lambda_1 * probabilities[(words[i])]
            P2 = lambda_2 * probabilities[(words[i-1],words[i])] + (1-lambda_2) * P1
            H += -math.log(P2,2)
            W += 1

print("Entropy="+str((H / W)))

            