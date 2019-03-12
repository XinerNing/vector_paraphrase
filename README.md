Paraphrase Identification using vector space models
---------------------------------------------------

This project examines vector space similarity for paraphrase identification.
It converts semantic textual similarity data to paraphrase identification data using threshholds.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

Instructor repository at <https://github.com/emmerkhofer/vector_paraphrase> . 

## Homework: `vector_pi.py`

* Train a logistic regression for PI on the training data using cosine similarity over
 two different vector spaces as features: word2vec and term frequency.
* Create a `SimilarityVectorizer` class that calculates the two similarity metrics.
* Set a `KeyedVectors` object and a `TfidfVectorizer` object as attributes to your `SimilarityVectorizer`
and use them in methods.
* Use the logistic regression implementation in `sklearn`. Use the word2vec tools in `gensim`.
* Print your model accuracy on dev.
* Update the `README.md` with a description of your code and your accuracy on dev.
* **HINT:** You may copy any relevant code and README material from previous homework.
* Grading uses 
1) your `vector_pi.py` code and 
2) your README.

This homework is submitted as a link to your Github repo on Canvas. You may submit the link early; 
we will grade the state of your repository at the deadline.


## 50K_GoogleNews_vecs.txt

A truncated version of the 300-vectors trained by Google Research on an enormous corpus of Google News.
Only the first 50K are circulated here to reduce memory and disk usage; 
the full file is available at <https://code.google.com/archive/p/word2vec/> .

## lab.py

`lab.py` calculates the pearsons correlation of word2vec vectors and the STS paraphrase dataset.
It creates sentence vectors by taking the mean of vectors for all in-vocabulary words.

Example usage:

`python lab.py --sts_data stsbenchmark/sts-dev.csv`

## vector_pi.py

TODO: Replace these instructions with a description of your code.
* Use `.md` filetype
* ~ 2 sentences about each similarity used.
* Describe what your script does.
* Include a usage example showing command line flags
* Describe your output.

## Results

TODO: paste your accuracy on dev here.

This system scores ______ . 
