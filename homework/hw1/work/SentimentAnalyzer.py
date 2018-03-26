__author__ = 'Taesoo Lim'

# Shared libraries
import os
import random
import re

# class NaiveBayesClassifierNLTK libraries
import nltk
from nltk.corpus import stopwords

from sklearn.datasets import load_svmlight_files
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

POLARITY_DATASET_PATH = os.path.join ('review_polarity', 'txt_sentoken')
POLARITY_POSITIVE_DATASET_PATH = os.path.join (POLARITY_DATASET_PATH, 'pos')
POLARITY_POSITIVE_REVIEWS = os.listdir (POLARITY_POSITIVE_DATASET_PATH)
POLARITY_NEGATIVE_DATASET_PATH = os.path.join (POLARITY_DATASET_PATH, 'neg')
POLARITY_NEGATIVE_REVIEWS = os.listdir (POLARITY_NEGATIVE_DATASET_PATH)

IMDB_DATASET_PATH = os.path.join ('aclImdb', 'train')
#IMDB_TEST_DATASET = os.path.join ('aclImdb', 'test', 'labeledBow.feat')
IMDB_POSITIVE_DATASET_PATH = os.path.join (IMDB_DATASET_PATH, 'pos')
IMDB_POSITIVE_REVIEWS = os.listdir (IMDB_POSITIVE_DATASET_PATH)
IMDB_NEGATIVE_DATASET_PATH = os.path.join (IMDB_DATASET_PATH, 'neg')
IMDB_NEGATIVE_REVIEWS = os.listdir (IMDB_NEGATIVE_DATASET_PATH)

#IBDB_DATAEST_PASH = os.path.join ()

"""
####################################################
                    CLASSES
####################################################
"""


class NaiveBayesClassifierNLTK:

    word_features = None

    def __init__ (self):
        pass

    def tokenize (self, sentence):
        """
        
        Delete Stopwords and Tokenize Sentence into words.
        
        """
        sentence = " ".join (re.findall (r"[\w']+|[./!?;]", sentence.rstrip()))

        tokens = nltk.word_tokenize (sentence)
        words = []
        english_stops = set (stopwords.words ('english'))

        # Filter feasible word set
        for token in tokens:
            if token not in english_stops:
                words.append (token)

        return words

    def build_corpus (self, document):
        """
        
            DOCUMENT_FORMAT      =>      CORPUS_FORMAT
        
        ('a sentences', 'label') => (['word1', 'word2'], 'label')
        
        """
        corpus = []
        for (sentence, label) in document:
            corpus.append ((self.tokenize (sentence), label))
        random.shuffle (corpus)

        return corpus


    def build_features (self, corpus):

        # Corpus should not be null
        assert corpus

        feature_words = []

        for (document, label) in corpus:
            feature_words.extend (document)

        word_freqdist = nltk.FreqDist (feature_words)
        return word_freqdist.keys ()

    def feature_function (self, document):
        """

        http://www.nltk.org/_modules/nltk/classify/util.html

        This function is to use nltk.classify.util.apply_feature ()
        feature function will be applied to each tokens

        """
        
        assert self.word_features

        words = set (document)
        features = {}

        for word in self.word_features:
            features['contains (%s)' % word] = (word in words)

        return features


    def train_nb (self, training_documents):
        """
        
        As required in the instruction for [HW1 - SentimentAnalysis], this function
        is to estimate the probabilities in the Naive Bayes model        
        
        """
        corpus = self.build_corpus (training_documents)
        assert corpus
        print ("COUPUS BUILD COMPLETED")

        self.word_features = self.build_features (corpus)
        assert self.word_features
        print ("FEATURES BUILD COMPLETED")

        training_set = nltk.classify.util.apply_features (self.feature_function, corpus)
        # print (training_set)

        classifier = nltk.NaiveBayesClassifier.train (training_set)
        assert classifier
        print ("TRAINING COMPLETED")

        return classifier

    def classify_nb (self, classifier_data, documents):
        """
        
        As required in the instruction for [HW1 - SentimentAnalysis], this function
        is to classify the document using the probabilities from training results.
        
        """

        # classifier must be defined
        assert classifier_data

        result = []
        
        for (document, label) in documents:
            guess = classifier_data.classify (self.feature_function (document.split ()))
            result.append (guess)

        return result


class SentimentAnalyzer:

    # data_choice = 0
    # ratio_choice = 0
    # fast_choice = 0

    def __init__ (self):
        self.data_choice = 0
        self.ratio_choice = 0
        self.fast_choice = 0

        self.documents = []

    def get_user_input (self):
        print ("\
####################################################\n\
    Which Dataset do you want to load?\n\
    \n\
    1. cornell (small)\n\
    2. stanford (large)\n\
    \n\
    WARNING: large dataset has not been tested...\n\
####################################################\n")

        user_input = input ("Dataset Choice [1 or 2] : ")
        assert user_input.isdigit ()

        self.data_choice = int (user_input)
        assert (self.data_choice == 1 or self.data_choice == 2)

        print("\n\
####################################################\n\
    Ratio of training_sets to test_sets\n\
    \n\
    1. ratio of 9 : 1 (Recommended)\n\
    2. ratio of 1 : 1 \n\
####################################################\n")

        user_input = input ("Ratio Choice [1 or 2] : ")
        assert user_input.isdigit ()

        self.ratio_choice = int (user_input)
        assert (self.ratio_choice == 1 or self.ratio_choice == 2)

        print("\n\
####################################################\n\
    Fast Running? (with 2000 training set and 100 test set For Testing)\n\
    \n\
    1. NO\n\
    2. YES\n\
####################################################\n")

        user_input = input ("Fast Running [1 or 2] : ")
        assert user_input.isdigit ()

        self.fast_choice = int (user_input)
        assert (self.fast_choice == 1 or self.fast_choice == 2)

    def build_documents (self):

        positive_documents = []
        negative_documents = []

        if self.data_choice == 1:
            positive_datafiles = POLARITY_POSITIVE_REVIEWS
            positive_dataset_path = POLARITY_POSITIVE_DATASET_PATH
            negative_datafiles = POLARITY_NEGATIVE_REVIEWS
            negative_dataset_path = POLARITY_NEGATIVE_DATASET_PATH

        elif self.data_choice == 2:
            positive_datafiles = IMDB_POSITIVE_REVIEWS
            positive_dataset_path = IMDB_POSITIVE_DATASET_PATH
            negative_datafiles = IMDB_NEGATIVE_REVIEWS
            negative_dataset_path = IMDB_NEGATIVE_DATASET_PATH

        else:
            raise NotImplementedError


        print ("\nLOADING Positive Reviews...\n")

        for review in positive_datafiles:
            for line in open (os.path.join (positive_dataset_path, review), 'r'):
                positive_documents.append ((line, 'pos'))

        print ("\nLOADING Negative Reviews...\n")

        for review in negative_datafiles:
            for line in open (os.path.join (negative_dataset_path, review), 'r'):
                negative_documents.append ((line, 'neg'))
    
        self.documents = positive_documents + negative_documents
        random.shuffle (self.documents)

    def evaluate (self, answer, prediction):
        positive_total = 0
        positive_correct = 0
        negative_total = 0
        negative_correct = 0

        positive_answer = []
        positive_prediction = []
        negative_answer = []
        negative_prediction = []

        for i in range (len (answer)):
            if answer[i] == 'pos':
                positive_total += 1
                positive_answer.append (answer[i])
                positive_prediction.append (prediction[i])

                if prediction[i] == 'pos':
                    positive_correct += 1
                elif prediction[i] == 'neg':
                    pass
                else:
                    raise NotImplementedError

            elif answer[i] == 'neg':
                negative_total += 1
                negative_answer.append (answer[i])
                negative_prediction.append (prediction[i])

                if prediction[i] == 'pos':
                    negative_correct += 1
                elif prediction[i] == 'neg':
                    pass
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        print ("\n")
        print ("Overall:")
        print ("accuracy : ", float (positive_correct + negative_correct) / float (positive_total + negative_total))
        # precision_recall_f1 = precision_recall_fscore_support (answer, prediction, average = 'weighted')
        print ("precision: ", precision_score (answer, prediction, average = 'weighted'))
        print ("recall: ", recall_score (answer, prediction, average = 'weighted'))
        print ("f1: ", f1_score (answer, prediction, average = 'weighted'))
        print ("\n")

        # print (positive_answer)
        # print (positive_prediction)
        # print (set (positive_answer))
        # print (set (positive_prediction))
        print ("pos category:")
        print ("total :", positive_total, ", correct :", positive_correct)
        print ("accuracy : ", float (positive_correct) / float (negative_total))
        # precision_recall_f1_pos = precision_recall_fscore_support (positive_answer, positive_prediction, average = 'weighted')
        print ("precision: ", precision_score (positive_answer, positive_prediction, average = 'weighted'))
        print ("recall: ", recall_score (positive_answer, positive_prediction, average = 'weighted'))
        print ("f1: ", f1_score (positive_answer, positive_prediction, average = 'weighted'))
        print ("\n")

        # print (negative_answer)
        # print (negative_prediction)
        print ("neg category:")
        print ("total :", negative_total, ", correct :", negative_correct)
        print ("accuracy : ", float (negative_correct) / float (negative_total))
        # precision_recall_f1_neg = precision_recall_fscore_support (negative_answer, negative_prediction, average = 'weighted')
        print ("precision: ", precision_score (negative_answer, negative_prediction, average = 'weighted'))
        print ("recall: ", recall_score (negative_answer, negative_prediction, average = 'weighted'))
        print ("f1: ", f1_score (negative_answer, negative_prediction, average = 'weighted'))
        print ("\n")


    def run (self):

        self.get_user_input ()

        self.build_documents ()

        training_documents = []
        testing_documents = []

        # Divide Training_sets and Test_sets
        total = float (len (self.documents))
        train_cutoff = int (round (total * 9 / 10))
        if (self.ratio_choice == 2):
            train_cutoff = int (round (total / 2))

        training_documents = self.documents [:train_cutoff]
        testing_documents = self.documents [train_cutoff:]

        # For Testing and Debugging
        if self.fast_choice == 2:
            training_documents = training_documents [:2000]
            testing_documents = testing_documents [:100]

        test_answer = []
        for (document, label) in testing_documents:
            test_answer.append (label)

        classifier = NaiveBayesClassifierNLTK ()

        # Task 1 : Train Document
        print ("Training the documents")
        classifier_trained = classifier.train_nb (training_documents)

        # Task 2 : Classify (test) Document
        print ("Testing the documents")
        prediction = classifier.classify_nb (classifier_trained, testing_documents)

        # Task 3 : Evaluate
        self.evaluate (test_answer, prediction)



"""
####################################################
                Actual Running Code
####################################################
"""
analyzer = SentimentAnalyzer ()
analyzer.run ()