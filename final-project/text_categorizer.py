import logging

import pandas as pd
from nltk.corpus import stopwords

from helpers import *

logging.basicConfig(level='INFO')
logger = logging.getLogger("Logger")

d_extractor = DataExtractor(logger)
m_extractor = MatrixExtractor()

f = open("data/train_data.txt", "r")
data = f.read()
logger.debug(data)

columns = ['subject', 'content', 'category']
text_columns = ['subject', 'content']
extracted_data = d_extractor.extract_data(data, text_columns)

#df, word_freq, word_2_vec, vocabulary = m_extractor.extract_matrices(extracted_data, columns, text_columns[1])

# KNN Classifier
success, error = KNNClassifier(extracted_data[:1000]).classify_all()

print("Accuracy in %:")
print(float(success) / (success + error) * 100.0)

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
