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
    toks = []
    return toks


def w2v_sentence(sent, word2vec):
    """Creates a sentence representation by taking the mean of all in-vocabulary word vectors.
    Returns None if no words are in vocabulary."""
    return None


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
    cos_sims.append(0)


pearson = pearsonr(cos_sims, dev_y)
print(f"word2vec pearsons: r={pearson[0]:.03}")
