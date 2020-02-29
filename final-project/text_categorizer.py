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


def run_1nn(train_data, test_data):
	knn_classifier = KNNClassifier(train_data)

	for doc in test_data:
		category = knn_classifier.classify(doc)
		print(category)


d_extractor = DataExtractor(logger)
m_extractor = MatrixExtractor()

train_data = extract_data(d_extractor, 'data/train_data_sample.txt')
test_data = extract_data(d_extractor, 'data/test_data_sample.txt')

print(train_data)
print(test_data)

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
