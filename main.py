import nltk
from nltk.corpus import stopwords # Required for cleaning the text
from scipy  import stats # Requried for calculating entropy
from math import log2


from string import ascii_lowercase
from collections import Counter

valid_letters = ascii_lowercase + "."

def map_dict(map_func, dict_to_map, map_keys=True):
	ret_dict = {}
	for key,value in dict_to_map.items():
		ret_dict[key] = map_func(key) if map_keys else map_func(value)
	return ret_dict

def letterCountList(txt):
	from collections import Counter
	from string import ascii_lowercase
	c=Counter(txt)
	ascii_lowercase += "."
	return [(x, c[x]) for x in ascii_lowercase]

def countDictToFrequencyDict(count_dict, corpus_size):
	return map_dict(lambda value: value / corpus_size, count_dict, False)

def lettersBigramCountList(txt):
	from collections import Counter
	from string import ascii_lowercase
	ascii_lowercase += "."
	bigrams = Counter(x+y for x, y in zip(*[txt[i:] for i in range(2)]))
	return [(bigram, count) for bigram, count in bigrams.most_common() if bigram[0] in ascii_lowercase and bigram[1] in ascii_lowercase]

def lettersBigramFrequencyDict(letters_bigram_vount_list):
	l = len(letter_count_dict)
	return map_dict(lambda letter_count: letter_count / l, letter_count_dict, False)
	
# calculate the kl divergence KL(P || Q)
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# calculate entropy H(P)
def entropy(p):
	return -sum([p[i] * log2(p[i]) for i in range(len(p))])
	
# calculate cross entropy H(P, Q)
def cross_entropy(p, q):
	return entropy(p) + kl_divergence(p, q)
	
def model_estimate(model):
	n = len(model)
	return -sum([log2(model[i]) for i in range(n)]) / n

# Loading the file
file = open("DevilsDictionary.txt", "r")
book = file.read()

# Cleaning the text
stop_words = stopwords.words('english')
clean_word_list = [word for word in book.split() if word not in stop_words]
txt = ' '.join(map(str,clean_word_list)).lower()
for letter in txt:
	if letter not in valid_letters:
		txt = txt.replace(letter, "")


corpus_size = len(txt)	

letters_count_list = letterCountList(txt)
letters, letters_count = zip(*letters_count_list)

letters_count_dict = dict(letters_count_list)

# Calculate a-z & '.' letters frequency in 1-gram model.
letters_frequency_dict = countDictToFrequencyDict(letters_count_dict, corpus_size)



# Calculate a-z & '.' letters frequency in 2-gram model.
letters_bigram_count_list = lettersBigramCountList(txt)
letters_bigram_count_dict = dict(letters_bigram_count_list)
letters_bigram_frequency_dict = countDictToFrequencyDict(letters_bigram_count_dict, corpus_size)



letters_frequencies = list(letters_frequency_dict.values())


print("1-grams letters model is a model of the english language that predict that an apeareance of a letter based on it's frequency (probability) in the corpus")
print("1-grams letters model is a model of the english language that predict that an apeareance of a letter based on it's relative frequency (probability) in the corpus --> given the previous letter what is the chance that the current letter will show up")
print("p is the actual probabilities of the english letters. p is unknown.")


q1_entropy = stats.entropy(letters_frequencies, base=2)
corpus_1_gram_frequencies = [letters_frequency_dict[letter] for letter in txt if letter in letters_frequency_dict]
p_estimate_based_on_1_gram = model_estimate(corpus_1_gram_frequencies)
print("\nEntropy of q1: " + str(q1_entropy))
print("\nEstimation of the entropy of p, based on q1 model: " + str(p_estimate_based_on_1_gram))
	

		
letters_conditional_bigram_frequency_dict = map_dict(lambda key: letters_bigram_count_dict[key] / letters_count_dict[key[0]], letters_bigram_count_dict)

letters_conditional_bigram_frequency_list = list(letters_conditional_bigram_frequency_dict.values())
q2_entropy = stats.entropy(letters_conditional_bigram_frequency_list, base=2)
p_estimate_based_on_q2 = model_estimate(letters_conditional_bigram_frequency_list)
print("\nq2_entropy: " + str(q2_entropy))
print("\nEstimation of the entropy of p, based on q2 model: " + str(p_estimate_based_on_q2))

