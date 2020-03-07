import logging

import pandas as pd
from nltk.corpus import stopwords

from helpers import *

logging.basicConfig(level='INFO')
logger = logging.getLogger("Logger")


def extract_data(d_extractor, path):
	f = open(path, "r")
	data = f.read()
	extracted_data = d_extractor.extract_data(data)
	return extracted_data


def output_categories(filename, categories):
	with open('data/' + filename, 'w') as f:
		for i in range(len(categories)):
			category = categories[i]
			f.write(category)

			if i < len(categories) - 1:
				f.write('\n')


def run_1nn(train_data, test_data):
	knn_classifier = KNNClassifier(train_data)

	categories = []
	i = 0
	for doc in test_data:
		print(i)
		i += 1
		category = knn_classifier.classify(doc)
		categories.append(category)

	output_categories('output1.txt', categories)


d_extractor = DataExtractor(logger)
m_extractor = MatrixExtractor()

train_data = extract_data(d_extractor, 'data/train_data.txt')
test_data = extract_data(d_extractor, 'data/test_data.txt')

run_1nn(train_data, test_data)

#columns = ['subject', 'content', 'category']
#text_columns = ['subject', 'content']
#df, word_freq, word_2_vec, vocabulary = m_extractor.extract_matrices(extracted_data, columns, text_columns[1])

#print(df)
# print('\n*****')
# print('data:')
# print('*****')
# print(data)

# print('\n**')
# print('df:')
# print('***')
# print(df[text_columns[1]])

# print('\n*********')
# print('wordfreq:')
# print('*********')
# print(word_freq)

# print('\n**********')
# print('word_2_vec:')
# print('***********')
# print(word_2_vec)

# print('\n**********')
# print('vocabulary:')
# print('***********')
# print(vocabulary)
