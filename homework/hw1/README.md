# [HW1] Sentiment Analysis (Classification Task)


## Development Environment Setting

This homework was implemented under the following environment
- Python 3.6.3
- Virtualenv 15.0.1

Additional packages can be downloaded from ```pip``` with using requirement.txt via following command

	$ pip install -r requirement.txt
    
This will install ```nltk 3.2.5``` and ```scikit-learn``` from ```pip```.

After Installing ```nltk``` and ```scikit-learn``` we have to install ```punkt```, ```stopwords``` in via nltk downloader.

	Python 3.6.3
	[GCC X.X.X xxxxx] on linux
	Type "help", "copyright", "credits" or "license" for more information.

	>>> import nltk
	>>> nltk.download('punkt')
	>>> nltk.download('stopwords')


We can download ```punkt``` via nltk downloader on ```python``` console.

## How to run

As long as the required packages are all installed, the only command to run this code is 

	$ python NaiveBayesClassifier.py
    
## Report
- In this homework, loading dataset from large file is not fully implemented (it can load it, but it doesn't get test set from provided test sets)
- Running with large dataset has never been fully tested.

## Available Dataset

- #### Movie Review Dataset (small)
	- [Dataset Download](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)
	- (Link : http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)
	- Size
		- 1,000 positive and 1,000 negative movie reviews from IMDB
	- Reference site: http://www.cs.cornell.edu/people/pabo/movie-review-data/

- #### Movie Reivew Dataset (large)
	- [Dataset Download](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
	- (Link : http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
	- Size
		- 25,000 training examples, 25,000 test examples
		- 80 MB compressed (Approximately 480 MB uncompressed)
	- Reference site: http://ai.stanford.edu/~amaas/data/sentiment


## Tool You Can Use

Here is a list of some python implementations of algorithms that you may find useful for your assignments.

- [scikit learn](http://scikit-learn.org/stable/)
- [NLTK](http://www.nltk.org/)

## Tasks
- #### Reading Dataset
	- Download and unpack the file provided above

- #### Training the Naive Bayes Classifier
	- Write a Python function that uses a training set of documents to estimate the probabilities in the Naive Bayes model. Return some data structure containing the probabilities. It could look something like this:

            def train_nb (training_documents):
                ...
                (return the data you need to classify new instances)
    
- #### Classifying new documents
	- Then write a Python function that classifies a new document. The inputs are (1) the probabilities returned by the first function; (2) the document to classify, which is a list of tokens.

            def classify_nb (classifier_data, document):
                ...
                (return the prediction of the classifier)

- #### Evaluating the classifier
Test your NB classifieir representations for each category (Pos, Neg) and report Precision, Recall, and F1 for each category using [scikit-learn](http://scikit-learn.org/stable/)

## References (Books)
- [Opinion Mining and Sentiment Analysis] by Bo Pang and Lillian Lee.
- [Introduction to Information Retrieval] by Christopher Manning, Prabhakar Raghavan, and Hinrich Schütze
- [Foundations of Statistical Natural Language Processing] by Christopher Manning and Hinrich Schuetze
