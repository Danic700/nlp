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


def read_categories(filename):
	with open('data/' + filename, 'r') as f:
		return f.read().split('\n')


def run_1nn(train_data, test_data, exclude_i=False):
	knn_classifier = KNNClassifier(train_data)

	categories = []
	for i in range(len(test_data)):
		doc = test_data[i]
		print(i)
		category = knn_classifier.classify(doc, i if exclude_i else None)
		categories.append(category)

	return categories


d_extractor = DataExtractor(logger)
m_extractor = MatrixExtractor()

train_data = extract_data(d_extractor, 'data/train_data.txt')
#test_data = extract_data(d_extractor, 'data/test_data.txt')

#run_1nn(train_data, test_data)
#output_categories('output1.txt', categories)

#categories = run_1nn(train_data, train_data, True)
#output_categories('output1_on_train_data.txt', categories)

class_categories = read_categories('output1_on_train_data.txt')
total = len(class_categories)
success = 0.0
for i in range(total):
	if class_categories[i] == train_data[i][2]:
		success += 1

print(success / total)

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
