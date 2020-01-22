def question5(txt, valid_letters, stop_words):
	
	# Internal imports
	import zipfslaw
	import text_cleaner

	# External imports
	from nltk.probability import FreqDist
	from nltk.corpus import stopwords

	num_of_top_results = 200000 # maximum number of words in the model
	num_of_top_results_display = 15 # number of words to display
		
	# Loading the file
	file = open("DevilsDictionary.txt", "r")
	book = file.read()

	print("\n++++++++++")
	print("==========")
	print("Question 5")
	print("==========")
	print("++++++++++\n")

	print("Investigating words characteristics.\n")
	print("\nGet tokens & word types:\n")
	
	# Clean the text.
	tokens, word_types = text_cleaner.clean_text(txt, stop_words, valid_letters)

	print("\n------------")
	print("Question 5.a")
	print("------------")

	print("number of tokens: " + str(len(tokens)))
	print("number of word types: " + str(len(word_types)))

	print("\n------------")
	print("Question 5.b")
	print("------------")
	print("\nGet words frequencies:\n")

	freq_dist = FreqDist(tokens) # Object which contains dictionary of word-->count

	print("\nTop " + str(num_of_top_results_display) + " words frequencies:")
	print("-------------------------")
	print(freq_dist.most_common(num_of_top_results_display))

	# Get zipfs table.
	letters_count_list = freq_dist.most_common(num_of_top_results)
	zipf_table = zipfslaw.create_zipf_table(letters_count_list)
	
	# Display zipfs law results in table & graph
	zipfslaw.print_zipf_table(zipf_table, num_of_top_results_display)
	zipfslaw.plot_zipfs_graph(zipf_table, freq_dist, num_of_top_results_display)