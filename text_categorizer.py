import logging
import matplotlib.pyplot as plt
import nltk

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score


from data_extractor import DataExtractor
from matrix_extractor import MatrixExtractor

def category_plot():
    fig = plt.figure(figsize=(8, 6))
    df.groupby('category').subject.count().plot.bar(ylim=0)
    plt.show()

def print_common_unigrams_bigrams():
    df['category_id'] = df['category'].factorize()[0]
    category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')

    labels = df.category
    features = tfidf.fit_transform(df.content).toarray()
    print(features.shape)
    N = 5
    Number = 1
    for category in df['category'].unique():
        features_chi2 = chi2(features, df['category'] == category)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [x for x in feature_names if len(x.split(' ')) == 1]
        bigrams = [x for x in feature_names if len(x.split(' ')) == 2]
        print(Number, "# '{}':".format(category))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        Number += 1

def split_train_test(rebalance):
    global count_vect, X_test, y_test, X_resampled, y_resampled, X_train, y_train
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(df['content'])
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)
    smote = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(counts, df['category'], test_size=1 / 5, random_state=54)

    if(rebalance==True):
        X_train, y_train = smote.fit_sample(X_train, y_train)

def run_naive_bayes():
    model = MultinomialNB().fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(accuracy_score(y_test, predicted))
    print(model.predict(count_vect.transform(["How do I tone my butt? What's the best exercise for butts?"])))


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

df = pd.DataFrame(extracted_data, columns=columns)

category_plot()
print_common_unigrams_bigrams()
split_train_test(False)  #True for rebalancing (training set) unbalanced data as seen in histogram
run_naive_bayes()

print('\n**')
print('df sample:')
print('***')
print(df.sample())


