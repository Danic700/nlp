from nltk.corpus import stopwords


# Loading the file
file = open("DevilsDictionary.txt", "r")
book = file.read()

# Cleaning the text
stop_words = stopwords.words('english')
clean_word_list = [word for word in book.split() if word not in stop_words]





print(clean_word_list)