

from collections import defaultdict

import pandas as pd
from nltk import word_tokenize

from gensim.models import Word2Vec


class MatrixExtractor:
        def vectorize_text(self, df, text_column):
                vectorized_text = []
                for sent in df[text_column]:
                        vectorized_text.append(Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1))
                return vectorized_text

        def extract_matrices(self, extracted_data, columns, text_column):

                df = pd.DataFrame(extracted_data, columns = columns)

                sentences = [row for row in df[text_column]]

                # Give index to each word
                vocabulary_index_dict = {}
                i = 0
                for sent in df[text_column]:
                        if len(sent) < 1:
                                continue

                        for word in sent:
                                vocabulary_index_dict[word] = i
                                i += 1

                word_freq = defaultdict(int)
                for sent in sentences:
                        if len(sent) < 1:
                                continue
                        for word in sent:
                                word_freq[vocabulary_index_dict[word]] += 1

                vocabulary = word_freq.keys()

                # x = df[text_column].values().tolist()
                # y = df['category'].values().tolist()

                x = list(df[text_column])
                y = list(df['category'])

                corpus = x + y

                tok_corp = [word_tokenize(' '.join(sent)) for sent in corpus]

                word_2_vec_model = Word2Vec(tok_corp, min_count=1, size = 32)

                print('\n&&&&&&&&&&')
                print('similarity')
                print('&&&&&&&&&&')
                print(word_2_vec_model.most_similar('hitting'))
                # self.vectorize_text(df, text_column)

                return (df, word_freq, word_2_vec_model, vocabulary)
