from nltk.tokenize import word_tokenize

punctuation_characters = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"

def _remove_unwanted_characters(text, chars_to_remove):
	"""
	Removes the unwanted characters from the text:
	Default characters to remove are: punctuation characters.
	"""
	tr = str.maketrans("", "", chars_to_remove)
	return text.translate(tr)
	
	
def clean_text(text, stop_words, valid_letters, chars_to_remove = punctuation_characters, join_spaces = False):
	"""
	Tokenize text, remove stop words, handle dot character(if necessary)
	Returns tuple of (tokens, word types)
	"""	
	
	clean_text = _remove_unwanted_characters(text, chars_to_remove)
	tokenized_words = word_tokenize(clean_text)
	filtered_words = [] # list of words (with duplicates) - the tokens of the corpus.
	filtered_words_set = set() # list unique of words - the word types.
	
	for word in tokenized_words:
		if _is_valid_word(word, stop_words, valid_letters):
			filtered_words.append(word)
			filtered_words_set.add(word)
	
	if join_spaces:
		return ' '.join(map(str,filtered_words)).lower()
	else:
		return (filtered_words, filtered_words_set)

def _is_valid_word(word, stop_words, valid_letters):
	if word in stop_words:
		return False
		
	for letter in word:
		if letter not in valid_letters:
			return False
			
	return True
