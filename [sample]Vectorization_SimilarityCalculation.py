import sys
import os,glob
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim import corpora, models, similarities
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.test.utils import datapath
from gensim.corpora import Dictionary
from gensim.test.utils import get_tmpfile

# NLTK Stop words
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

%matplotlib inline
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Load a potentially pretrained model from disk.
lda_model =  models.LdaModel.load('lda_model')

# Load previous dictionary
id2word = Dictionary.load_from_text('/Users/hellofutrue/Desktop/Insight/Python/Feb/dictionary')

posts_influencers = pd.read_csv('/Users/hellofutrue/Desktop/Insight/Python/Feb/files/posts_influencers.csv')
posts_influencers = posts_influencers.rename(index=str, columns={'Unnamed: 0': "people", '0': 'content'})
data = posts_influencers.content.values.tolist()

def preprocessing(dat):
    # Tokenization
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    
    data_words = list(sent_to_words(dat))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    stop_words = stopwords.words('english')
    stop_words.extend(['com', 'bio','link','get','go'])
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    other_texts = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    return(other_texts)
    
vecs = []
for i in range(0, len(data)):
    other_texts = preprocessing(data[i].split("\',"))
    other_corpus = [id2word.doc2bow(text) for text in other_texts]
    unseen_doc = other_corpus[0]
    vector = lda_model[unseen_doc]
    vecs.append(vector[0])  
    
topicn = 6
topic_name = ['Topic 1','Topic 2','Topic 3','Topic 4','Topic 5','Topic 6']

vector = pd.DataFrame()
for i in range(0, topicn):
    p = [item[i] for item in vecs]
    a = [item[1] for item in p]
    df = pd.DataFrame(np.array(a).reshape(1,len(a)))
    vector = vector.append(df)

vector = vector.T
vector.columns = list(topic_name)
vector.index = list(posts_influencers['people'])

vector.to_csv('/Users/hellofutrue/Desktop/Insight/Python/Feb/files/vector.csv')

def dst(nameinput):
    dst = pd.DataFrame()
    for people in vector.index:
        row = vector.loc[nameinput, : ].values
        other = vector.loc[people, : ].values
        score = pd.DataFrame(distance.euclidean(row, other),index=[people], columns=[nameinput])
        dst = dst.append(score)
    result = dst.sort_values([nameinput], ascending=[1])[1:6]
    return result
    
result = dst('therock')
