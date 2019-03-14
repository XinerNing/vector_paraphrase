# coding: utf-8
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


def preprocess_text(text):
    """Preprocess one sentence: tokenizes, lowercases,
     removes punctuation tokens and stopwords.
     Returns a list of tokens."""
    remove_tokens = set(stopwords.words("english") + list(string.punctuation))
    toks = word_tokenize(text)
    toks = [tok for tok in toks if tok not in remove_tokens]
    return toks


def w2v_sentence(sent, word2vec):
    """Creates a sentence representation by taking the mean of all in-vocabulary word vectors.
    Returns None if no words are in vocabulary."""
    toks = preprocess_text(sent)
    veclist = [word2vec[tok] for tok in toks if tok in word2vec]
    if len(veclist) == 0:
        return None
    #vec_mat = np.vstack(veclist)
    mean_vec = np.mean(veclist, axis=0)
    return mean_vec


def load_sts(sts_data):
    # read the dataset
    texts = []
    labels = []

    with open(sts_data, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1,t2))

    return texts, np.asarray(labels)


sts_dir = "../sts_strings/stsbenchmark"
sts_dev = f"{sts_dir}/sts-dev.csv"

w2v_file = "50K_GoogleNews_vecs.txt"

# load the texts
dev_texts, dev_y = load_sts(sts_dev)

# load word2vec
w2v_vecs = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

# get cosine similarities of every pair in dev
# if either sentence is completely out of vocabulary, record "0" as the similarity
cos_sims = []
for t1,t2 in dev_texts:
    t1_vector = w2v_sentence(t1, w2v_vecs)
    if t1_vector is None:
        cos_sims.append(0)
        continue
    t1_vector = t1_vector.reshape((1, -1)) # shape for cosine similarity
    t2_vector = w2v_sentence(t2, w2v_vecs)
    if t2_vector is None:
        cos_sims.append(0)
        continue
    t2_vector = t2_vector.reshape((1, -1))
    pair_similarity = cosine_similarity(t1_vector, t2_vector)[0, 0]
    cos_sims.append(pair_similarity)


pearson = pearsonr(cos_sims, dev_y)
print(f"word2vec pearsons: r={pearson[0]:.03}")
