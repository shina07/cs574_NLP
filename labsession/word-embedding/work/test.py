from nltk.corpus import brown

sentences_brown = brown.sents()

for i in range(5):
    print str(i + 1) + ')' + ''.join(sentences_brown[i])


print 'training...',

# import Word2Vec library
from gensim.models.word2vec import Word2Vec

# train a word embedding model with Brown corpus
model_brown = Word2Vec (sentences_brown, size = 100, window = 10, min_count = 50, workers = 100)

print "done!"
print "\n"

print model_brown['power']

print 'plotting...',

from slkearn.manifold import TSNE

# extract all word vectors
X = model_brown [model_brown.wv.vocab]

# reduce the dimension of word vectors from 100-0 to 3-D

