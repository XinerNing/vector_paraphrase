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

Download [here](https://drive.google.com/file/d/1VKz_8FFTQebHIL-Ok_Qo63rwhR6dbu4G/view?usp=sharing).

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

* cosine similarity of word2vec: Based on the 50K_GoogleNews_vecs.txt, each sentence is converted to a vector of 300 numbers. Then the two vectors are compared by the cosine similarity. 
* cosine similarity of term frequency: Each sentence is convert into a vector of numbers based on its words frequences across the text. Then the two vectors are compared by the cosine similarity. 
* Script description: This script creates a matrix with two features: cosine similarity of word2vec and term frequency. The matrix is then being used to train a regression model to identify paraphrase. A model accuracy score is returned as the result.
* Include a usage example showing command line flags: python vector_pi.py (50K_GoogleNews_vecs.txt is too large to be uploaded to github repository. So in order to run the .py, the environment should have this file downloaded.)
* Output description: Accuracy of the regression model is 0.85, showing that the regression model does a fairly good job in identifying paraphrase.

## Results

TODO: paste your accuracy on dev here.

This system scores 0.8468468468468469 . 
