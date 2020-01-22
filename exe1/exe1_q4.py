
def question4(txt, valid_letters, stop_words):

	# Internal imports
	import text_cleaner
	
	# External imports
	from scipy  import stats # Requried for calculating entropy
	from math import log2
	from collections import Counter

	def map_dict(map_func, dict_to_map, map_keys=True):
		ret_dict = {}
		for key,value in dict_to_map.items():
			ret_dict[key] = map_func(key) if map_keys else map_func(value)
		return ret_dict

	def letterCountList(txt):
		from collections import Counter
		c=Counter(txt)
		return [(x, c[x]) for x in valid_letters]

	def countDictToFrequencyDict(count_dict, corpus_size):
		return map_dict(lambda value: value / corpus_size, count_dict, False)

	def lettersBigramCountList(txt):
		from collections import Counter
		bigrams = Counter(x+y for x, y in zip(*[txt[i:] for i in range(2)]))
		return [(bigram, count) for bigram, count in bigrams.most_common() if bigram[0] in valid_letters and bigram[1] in valid_letters]
		
	def model_estimate(model):
		n = len(model)
		return -sum([log2(model[i]) for i in range(n)]) / n

	txt = text_cleaner.clean_text(txt, stop_words, valid_letters, join_spaces = True)
	corpus_size = len(txt)

	print('1-gram corpus size: ' + str(corpus_size))

	letters_count_list = letterCountList(txt)
	letters, letters_count = zip(*letters_count_list)
	letters_count_dict = dict(letters_count_list)

	# Calculate a-z & '.' letters frequency in 1-gram model.
	letters_frequency_dict = countDictToFrequencyDict(letters_count_dict, corpus_size)

	# Calculate a-z & '.' letters frequency in 2-gram model.
	letters_bigram_count_list = lettersBigramCountList(txt)
	letters_bigram_count_dict = dict(letters_bigram_count_list)	
	bigram_corpus_size = sum(letters_bigram_count_dict.values())
	
	print('2-gram corpus size: ' + str(bigram_corpus_size))
	
	letters_frequencies = list(letters_frequency_dict.values())

	print("1-grams letters model is a model of the english language that predict that an apeareance of a letter based on it's frequency (probability) in the corpus")
	print("1-grams letters model is a model of the english language that predict that an apeareance of a letter based on it's relative frequency (probability) in the corpus --> given the previous letter what is the chance that the current letter will show up")
	print("p is the actual probabilities of the english letters. p is unknown.")
	print("Therefore instead of calculating cross entropy directly,")
	print("the cross entropy will be estimated based on Monte Carlo method.")

	q1_entropy = stats.entropy(letters_frequencies, base=2)
	corpus_1_gram_frequencies = [letters_frequency_dict[letter] for letter in txt if letter in letters_frequency_dict]
	
	letters_count_dict_size = len(letters_count_dict)
	print('letters_count_dict_size = ' + str(letters_count_dict_size))
	cross_entropy_estimation_p_q1 = model_estimate(corpus_1_gram_frequencies)
	print("\nEntropy of q1: " + str(q1_entropy))
	print("\nMonte Carlo cross entropy estimation of p and q1: " + str(cross_entropy_estimation_p_q1))

	letters_conditional_bigram_frequency_dict = map_dict(lambda key: letters_bigram_count_dict[key] / (letters_count_dict[key[1]] * letters_count_dict_size), letters_bigram_count_dict)		
	letters_conditional_bigram_frequency_list = list(letters_conditional_bigram_frequency_dict.values())
	q2_entropy = stats.entropy(letters_conditional_bigram_frequency_list, base=2)
	
	cross_entropy_estimation_p_q2 = model_estimate(letters_conditional_bigram_frequency_list)
	print("\nq2_entropy: " + str(q2_entropy))
	print("\nMonte Carlo cross entropy estimation of p and q2: " + str(cross_entropy_estimation_p_q2))
