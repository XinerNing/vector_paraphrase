import numpy as np
import argparse

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer



class SimilarityVectorizer:
    """Creates a vector of similarities for pairs of sentences"""

    def __init__(self, tfidf_vectorizer, word2vec):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.word2vec = word2vec

    def tfidf_sim(self, t1, t2):
        """Returns a float of cosine similarity between tfidf vectors for two sentences.
        Uses preprocessing including stemming."""

        return 0

    def w2v_sim(self, t1, t2):
        """Returns a float of cosine similarity between w2v vectors for two sentences.
        w2v vectors are the mean of any in-vocabulary words in the sentence, after lowercasing.
        Cosine similarity is 0 if either sentence is completely out of vocabulary. """

        return 0

    def load_X(self, sent_pairs):
        """Create a matrix where every row is a pair of sentences and every column in a feature.
        """
        X = np.zeros((len(sent_pairs), 2))

        return X


def preprocess_text(text, stem=False):
    """Preprocess one sentence: tokenizes, lowercases, (optionally) applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace."""
    toks = []
    return toks


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


def sts_to_pi(texts, labels, min_paraphrase=4.0, max_nonparaphrase=3.0):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_nonparaphrase."""

    # get only rows in the right intervals
    pi_rows = np.where(np.logical_or(labels>=min_paraphrase, labels<=max_nonparaphrase))[0]

    pi_texts = [texts[i] for i in pi_rows]
    # using indexing to get the right rows out of labels
    pi_y = labels[pi_rows]
    # convert to binary using threshold
    pi_y = pi_y > max_nonparaphrase
    return pi_texts, pi_y


def main(sts_train_file, sts_dev_file, w2v_file):
    """Fits a logistic regression for paraphrase identification, using string similarity metrics as features.
    Prints accuracy on held-out data. Data is formatted as in the STS benchmark"""

    # load word2vec
    w2v_vecs = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

    # loading train
    train_texts_sts, train_y_sts = load_sts(sts_train_file)
    train_texts, train_y = sts_to_pi(train_texts_sts, train_y_sts)

    # prepare tfidf vectorizer using train
    tfidf_vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word",
      token_pattern="\S+", use_idf=True, min_df=10)

    # create a SimilarityVectorizer object
    sim_vectorizer = SimilarityVectorizer(tfidf_vectorizer, w2v_vecs)

    print("Calculating similarities for train")
    train_X = sim_vectorizer.load_X(train_texts)

    # loading dev
    dev_texts_sts, dev_y_sts = load_sts(sts_dev_file)
    dev_texts, dev_y = sts_to_pi(dev_texts_sts, dev_y_sts)

    print("Calculating similarities for dev")
    dev_X = sim_vectorizer.load_X(dev_texts)

    print("Fitting and evaluating model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="../sts_strings/stsbenchmark/sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="../sts_strings/stsbenchmark/sts-train.csv",
                        help="train file")
    parser.add_argument("--w2v_file", type=str, default="50K_GoogleNews_vecs.txt",
                        help="file with word2vec vectors as text")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file, args.w2v_file)
