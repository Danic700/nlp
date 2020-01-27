

from gensim.models import Word2Vec
from collections import defaultdict
import pandas as pd


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

                word_2_vec = self.vectorize_text(df, text_column)

                return (df, word_freq, word_2_vec, vocabulary)