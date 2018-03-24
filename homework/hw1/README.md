# [HW1] Sentiment Analysis (Classification Task)


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
