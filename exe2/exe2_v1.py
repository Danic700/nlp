from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk import bigrams, trigrams
from collections import Counter, defaultdict



from text_cleaner import clean_text

# Loading the file
file = open("corpus_combined/persuasion.txt", "r")
book = file.read()
file.close()

# Get stop words.
stop_words = set(stopwords.words("english"))

(filtered_words, filtered_words_set, clean_text) = clean_text(book, stop_words)



# Create a placeholder for trigram model
bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
trigram_model = defaultdict(lambda: defaultdict(lambda: 0))
 
sent_text = sent_tokenize(clean_text)
for sentence in list(sent_text):
    tokens = word_tokenize(sentence)
    for w1, w2 in bigrams(tokens, pad_right=True, pad_left=True):
        bigram_model[(w1)][w2] += 1

    for w1, w2, w3 in trigrams(tokens, pad_right=True, pad_left=True):
        trigram_model[(w1, w2)][w3] += 1


# Let's transform the counts to probabilities
for w1 in bigram_model:
    total_bigram_count = float(sum(bigram_model[w1].values()))
    for w1 in bigram_model[w1]:
        bigram_model[w1][w2] /= total_bigram_count

for w1_w2 in trigram_model:
    total_count = float(sum(trigram_model[w1_w2].values()))
    for w3 in trigram_model[w1_w2]:
        trigram_model[w1_w2][w3] /= total_count


# for  w1 in bigram_model:
#     if w1 is not None and len(dict(bigram_model[w1])) > 1000:
#         print(w1, dict(bigram_model[w1]))

# for  w1, w2 in trigram_model:
#     if w1 is not None and w2 is not None and len(dict(trigram_model[w1, w2])) > 100:
#         print(w1, w2, dict(trigram_model[w1, w2]))


print('###############################################################################################################################################')




def good_turing(tokens):
    N = len(tokens)
    C = Counter(tokens)    
    N_c = Counter(list(C.values()))
    assert(N == sum([k * v for k, v in N_c.items()]))
    default_value = N_c[1] / N
    model = defaultdict(lambda: default_value)
    types = C.keys()
    B = len(types)  
    for _type in types:
        c = C[_type]
        model[_type] = (c + 1) * N_c[c + 1] / (N_c[c] * N)
    return model

def test_good_turing(tokens):
    N = len(tokens)
    print(N)
    C = Counter(tokens)
    print(C.values())  


#good_turing_unigram_model = good_turing(word_tokenize(clean_text))

#good_turing_bigram_model = good_turing(bigram_model)

test_good_turing(bigram_model)

#print(good_turing_bigram_model)