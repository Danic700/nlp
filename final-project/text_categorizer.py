import logging
import pandas as pd
from nltk.corpus import stopwords



from data_extractor import DataExtractor
from matrix_extractor import MatrixExtractor

logging.basicConfig(level = 'INFO')
logger = logging.getLogger("Logger")

data_extractor = DataExtractor(logger)
matrix_extractor = MatrixExtractor()

f = open("train_data.txt", "r")
data = f.read()
logger.debug(data)

columns = ['subject', 'content', 'category']
text_columns = ['subject', 'content']
extracted_data = data_extractor.extract_data(data, text_columns)

df, word_freq, word_2_vec, vocabulary = matrix_extractor.extract_matrices(extracted_data, columns, text_columns[1])


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
