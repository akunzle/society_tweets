import csv

# Define file paths
input_file = '/Users/andreakunzle/.pyenv/versions/3.10.0/tweets/algoritmo_tweets.csv'
output_file = '/Users/andreakunzle/.pyenv/versions/3.10.0/sentiment_analysis/algoritmo_tweets.csv'
sentiment_labels = ['Negative', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Positive', 'Negative', 'Positive', 'Neutral', 'Negative', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Negative', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Neutral', 'Positive', 'Negative', 'Negative', 'Negative', 'Neutral', 'Positive', 'Neutral', 'Positive', 'Positive']

# Filter out rows starting with 'Replying to' and empty rows
with open(input_file, 'r', newline='', encoding='utf-8') as file, \
     open(output_file, 'w', newline='', encoding='utf-8') as out_file:

    reader = csv.DictReader(file)
    writer = csv.writer(out_file)
    writer.writerow(['postDate', 'responding'])  # Write the new header row

    for row in reader:
        if row['responding'].strip() and not row['responding'].startswith('Replying to'):
            writer.writerow([row['postDate'], row['responding'].replace('\n', '')])

# Read the filtered file
with open(output_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    filtered_tweets = [row for row in reader]
    print(len(filtered_tweets))

# Add new column with sentiment labels
for i in range(1, len(filtered_tweets)):
    filtered_tweets[i].append(sentiment_labels[i-1])

# Write to file with new column and delete old header
with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['postDate', 'responding', 'sentiment'])  # Write the new header row
    for row in filtered_tweets[1:]:  # Skip the first row (old header)
        writer.writerow(row)

with open(output_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    algoritmo_tweets = [row for row in reader]
    print(algoritmo_tweets)


#Start cleaning the tweets

import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load the NLTK stop words in Spanish
stop_words = set(stopwords.words('spanish'))

# Define a function to clean the tweets
def clean_tweet(tweet):

    # Remove URLs and web links
    tweet = re.sub(r'http\S+', '', tweet)

    # Add space to tweets that require it
    tweet = re.sub(r'\.(?=[^\s\d])', '. ', tweet)

    # Remove mentions
    tweet = re.sub(r'@\S+', '', tweet)

    # Remove numbers
    tweet = re.sub(r'[0-9]+', '', tweet)

    # Remove punctuation and special characters (except for '#')
    tweet = tweet.translate(str.maketrans('', '', string.punctuation.replace('#', '')+ '¿'))

    # Remove extra white spaces
    tweet = re.sub(r'\s+', ' ', tweet)

    return tweet


# Clean the tweets
cleaned_tweets = []
for tweet in filtered_tweets:
    cleaned_tweet = clean_tweet(tweet[1])  # assuming the tweet text is in the first column
    cleaned_tweets.append(cleaned_tweet)

# Print the cleaned tweets
print(cleaned_tweets)

from nltk.tokenize import word_tokenize

# Tokenize the cleaned tweets
tokenized_tweets = [word_tokenize(tweet) for tweet in cleaned_tweets]
print(tokenized_tweets)

#Clean the list of the tweets tokens 
import re
import string

def remove_punctuations_and_spaces(tokens):
    # Remove extra white spaces and punctuation
    cleaned_tokens = []
    for tweet_tokens in tokens:
        cleaned_tweet_tokens = []
        for token in tweet_tokens:
            token = re.sub(r'\s+', ' ', token)
            token = re.sub(r'”', ' ', token)
            token = re.sub(r'“', ' ', token)
            token = token.translate(str.maketrans('', '', string.punctuation))
            if token.strip():
                cleaned_tweet_tokens.append(token)
        cleaned_tokens.append(cleaned_tweet_tokens)
    return cleaned_tokens

cleaned_tokens = remove_punctuations_and_spaces(tokenized_tweets)
print(cleaned_tokens)

#Normalization
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Normalize the tokens using PorterStemmer
normalized_tweets = [[stemmer.stem(word.lower()) for word in tweet] for tweet in cleaned_tokens]
print(normalized_tweets)

# Remove stop words from the tokens
filtered_tweets = [[word for word in tweet if word.lower() not in stop_words] for tweet in normalized_tweets]
print(filtered_tweets)

# Load a pre-trained sentiment analysis model
import nltk

nltk.download('all-corpora')

import csv

sentiment_labels = ['Negative', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Positive', 'Negative', 'Positive', 'Neutral', 'Negative', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Negative', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Neutral', 'Positive', 'Negative', 'Negative', 'Negative', 'Neutral', 'Positive', 'Neutral', 'Positive', 'Positive']
tweets = [' '.join(tweet) for tweet in filtered_tweets[1:]] # list of tweets


with open('/Users/andreakunzle/.pyenv/versions/3.10.0/sentiment_analysis/sentiment_labels.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tweet', 'Sentiment Label']) # write the header row
    for i in range(len(sentiment_labels)):
        writer.writerow([tweets[i], sentiment_labels[i]]) # write each row of tweet and corresponding label


from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
import csv
import pickle
from sklearn.model_selection import KFold

# Load the labeled tweets from the CSV file
with open('/Users/andreakunzle/.pyenv/versions/3.10.0/sentiment_analysis/sentiment_labels.csv', 'r') as file:
    reader = csv.reader(file)
    labeled_tweets = [(row[0], row[1]) for row in reader]

# Split the labeled tweets into training and validation sets
train_set, val_set = train_test_split(labeled_tweets, test_size=0.2)

# Define a function to extract features from the tweets
def extract_features(tweet):
    return {'Tweet': tweet}

# Tune the smoothing factor
for alpha in [0.5, 1, 1.5, 2]:
    # Train the classifier using the training set
    cl = NaiveBayesClassifier(train_set, feature_extractor=extract_features, alpha=alpha)

    # Evaluate the classifier on the validation set
    accuracy = cl.accuracy(val_set)
    print(f'Smoothing Factor: {alpha} - Accuracy: {accuracy}')

# Perform 5-fold cross-validation
kfold = KFold(n_splits=5)
fold = 0
for train_indices, val_indices in kfold.split(labeled_tweets):
    fold += 1
    print(f'Fold {fold}:')

    # Split the data into training and validation sets
    train_set = [labeled_tweets[i] for i in train_indices]
    val_set = [labeled_tweets[i] for i in val_indices]

    # Train the classifier using the training set
    cl = NaiveBayesClassifier(train_set, feature_extractor=extract_features)

    # Evaluate the classifier on the validation set
    accuracy = cl.accuracy(val_set)
    print(f'Accuracy: {accuracy}')

    # Save the classifier to a file using pickle
    with open(f'sentiment_classifier_fold{fold}.pkl', 'wb') as file:
        pickle.dump(cl, file)


from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Train and evaluate SVM
svm = SklearnClassifier(SVC())
svm.train([(extract_features(tweet), label) for tweet, label in train_set])
svm_accuracy = nltk.classify.accuracy(svm, [(extract_features(tweet), label) for tweet, label in val_set])
print(f'SVM Accuracy: {svm_accuracy}')

# Train and evaluate logistic regression
lr = SklearnClassifier(LogisticRegression())
lr.train([(extract_features(tweet), label) for tweet, label in train_set])
lr_accuracy = nltk.classify.accuracy(lr, [(extract_features(tweet), label) for tweet, label in val_set])
print(f'Logistic Regression Accuracy: {lr_accuracy}')

# Train and evaluate decision tree
dt = SklearnClassifier(DecisionTreeClassifier())
dt.train([(extract_features(tweet), label) for tweet, label in train_set])
dt_accuracy = nltk.classify.accuracy(dt, [(extract_features(tweet), label) for tweet, label in val_set])
print(f'Decision Tree Accuracy: {dt_accuracy}')



























"""
@author: jorgesaldivar
"""

import re
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from googletrans import Translator
from time import sleep 


def download_stop_words():
    # Downloading English stopwords
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


with open(output_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    algoritmo_tweets = [row for row in reader]
    print(algoritmo_tweets)

with open(output_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    # Skip the header row
    next(reader)
    # Iterate through each row and save the 'responding' text
    tweet_texts = []
    for row in reader:
        tweet_texts.append(row[1])
    print(tweet_texts)

'''
Based on http://brandonrose.org/clustering
'''

def tokenize_and_remove_stop_words(text, specific_words_to_delete=[], 
                                   join_words=False, language='english'):
    # define stop words
    stop_words = nltk.corpus.stopwords.words(language) + ['.', ',', '--', 
                                        '\'s', '?', ')', '(', ':', '\'', 
                                        '\'re', '"', '-', '}', '{', u'—']
    # first tokenize by sentence, then by word to ensure that punctuation 
    # is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in 
              nltk.word_tokenize(sent)]
    # removing stop words
    cleaned_tokens = [word for word in tokens if word not in 
                      set(stop_words)]
    # keep only letter
    alpha_tokens = cleaned_tokens
    filtered_tokens = []
    for token in alpha_tokens:
        if token not in specific_words_to_delete:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token.strip())
    if join_words:
        return ' '.join(filtered_tokens)
    else:
        return filtered_tokens


tweets_stop_words_removed = [tokenize_and_remove_stop_words(tweet, language = 'spanish') for tweet in tweet_texts]
print(tweets_stop_words_removed)


def tokenize_and_stem(text, specific_words_to_delete=[], 
                      join_words=False, language='english'):
    # define stop words
    stop_words = nltk.corpus.stopwords.words(language) + [ '.', ',', '--', 
                                        '\'s', '?', ')', '(', ':', '\'', 
                                        '\'re', '"', '-', '}', '{', u'—', ]
    # first tokenize by sentence, then by word to ensure that punctuation 
    # is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in 
              nltk.word_tokenize(sent)]
    # removing stop words
    cleaned_tokens = [word for word in tokens if word not in 
                      set(stop_words)]
    # keep only letter
    alpha_tokens = [re.sub('[^A-Za-z]', ' ', token) for token in cleaned_tokens]
    filtered_tokens = []
    for token in alpha_tokens:
        if token not in specific_words_to_delete:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token.strip())
    # stemming
    stemmer = SnowballStemmer(language)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    if join_words:
        return ' '.join(stems)
    else:
        return stems
    

tweets_stemmed = [tokenize_and_stem(tweet, language = 'spanish') for tweet in tweet_texts]
print(tweets_stemmed)

def clean_html_tags(raw_html):
    return BeautifulSoup(raw_html, "lxml").text

def shuffled(x):
    y = x[:]
    random.shuffle(y)
    return y

def clean_emojis(doc):
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', doc)

def translate_doc(doc, src="es", dest="en"):
    translator = Translator()
    while(True):
        try:
            t = translator.translate(doc[0:4999], src=src, dest=dest).text
            return t
        except:
            sleep(1)

def mark_negation_es(text, join_words=False):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in 
              nltk.word_tokenize(sent)]
    negations = set(["no", "nunca", "nada", "nadie", 
                    "ninguno", "ningún", "ninguna"])
    stop = set([ '.', ',', ';', '!', '?', ')', '}', 'y'])

    marked = []
    negate = False
    for token in tokens:
        if negate and token in stop:
            negate = False
        if negate:
            marked.append(token + "_NEG")
        else:
            marked.append(token)
        if token in negations:
            negate = True
    if join_words:
        return ' '.join(marked)
    else:
        return marked
    

tweets_mark_negation = [mark_negation_es(tweet) for tweet in tweet_texts]
print(tweets_mark_negation)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
from cca_core.utils import tokenize_and_remove_stop_words, tokenize_and_stem, \
                  clean_emojis, translate_doc, mark_negation_es

class SentimentAnalyzer:
    """
    Analyzes the sentiment polarity of a collection of documents.
    It determines wether the feeling about each doc is positive,
    negative or neutral

    Parameters
    ----------
    neu_inf_lim : float, -0.3 by default
        If a doc's polarity score is lower than this paramenter,
        then the sentiment is considered negative.
        Use values greater than -1 and lower than 0.

    neu_sup_lim : float, 0.3 by default
        If a doc's polarity score is greater than this parameter,
        then the sentiment is considered positive.
        Use values greater than 0 and lower than 1.

    language: string, 'english'; by default
        Language on which documents are written
        There are 2 languages supported natively:
        1 - 'english': through the ntlk_vader algorithms
        2 - 'spanish': through the ML_SentiCon algorithm
        If you use another language, the module will first translate each 
        document to english (using Google Translate AJAX API), so it can later
        re-use ntlk_vader algorithm for english docs.
    """

    _sia = SentimentIntensityAnalyzer()
    _tagged_docs = []

    def __init__(self, neu_inf_lim=-0.05,
                 neu_sup_lim=0.05,
                 language="english",
                 negation_handling = False,
                 hashtags=[],
                 n_gram_handling = True):
        self.neu_inf_lim = neu_inf_lim
        self.neu_sup_lim = neu_sup_lim
        self.language=language
        self.negation_handling = negation_handling
        self.n_gram_handling = n_gram_handling
        self.hashtags = hashtags
        self.translate = False
        self.need_normalization = False
        self.mlsent = {}
        self.spa_lemmas = []
        self.min_score = 100
        self.max_score = -100
        if language == "spanish":
            self.load_spa_resources()
            self.algorithm = "ML-Senticon"
            # More research needed on normalizing scores
            # self.need_normalization = True
        elif language == "english":
            self.algorithm = "nltk_vader"
        else:
            self.algorithm = "nltk_vader"
            self.translate = True
            if self.language == "french":
                self.src_lang = "fr"
            elif self.language == "portuguese":
                self.src_lang = "pt"


    def load_spa_resources(self):
        """
        Load lexicons required when analyzing text in spanish.
        - Michal Boleslav Měchura's Spanish lemmatization dictionary.
        - ML SentiCon: Cruz, Fermín L., José A. Troyano, Beatriz Pontes, 
          F. Javier Ortega. Building layered, multilingual sentiment 
          lexicons at synset and lemma levels, 
          Expert Systems with Applications, 2014.
        
        """
        dir = os.path.dirname(os.path.realpath(__name__)) + "/env/lib/python3.9/site-packages/cca_core/lexicon_lib"
        fl = open(dir+"/lemmatization-es.txt")
        lines = fl.readlines()
        self.spa_lemmas = [(l.split()[0], l.split()[1]) for l in lines]
        fl.close()
        fmd = open(dir+"/MLsenticon.es.xml",  encoding='utf-8')
        for l in fmd.readlines():
            sl = l.split()
            if len(sl) == 6:
                word = sl[4].replace("_", " ")
                pol = float(sl[2].split('"')[1])
                self.mlsent[word] = pol
        fmd.close()
        # Load new words found on political tweets
        fnpw = open(dir+"/new_pos_words.txt", encoding='utf-8')
        for w in fnpw.readlines():
            word = w.replace("\n", "")
            self.mlsent[word] = 0.35
        fnpw.close()

        fnnw = open(dir+"/new_neg_words.txt", encoding='utf-8')
        for w in fnnw.readlines():
            word = w.replace("\n", "")
            self.mlsent[word] = -0.39
        fnnw.close()
        # # load hashtags as positives words for lexicon
        # for h in self.hashtags:
        #     word = h.replace("#", "").lower()
        #     self.mlsent[word] = 1.0
        #     print("new hashtags", word)

       

    def lemmatize_spa(self, spa_word):
        """
        Return spanish lemma for a given word

        Parameters
        ----------
        spa_word : string
            Spanish word to lemmatize
        """
        # spa_word is a word form
        res1 = [i[0] for i in self.spa_lemmas if i[1]==spa_word]
        if len(res1)==1:
            return res1[0]
        # spa_word is already a lemma
        res2 = [i[0] for i in self.spa_lemmas if i[0]==spa_word]
        if len(res2)>0:
            return res2[0]
        return ""


    def spa_polarity_score(self, doc):
        """
        Calculate a polarity score for a given doc usin ML-Senticon

        Parameters
        ----------
        doc : string
            Text to score

        Returns
        -------
        mlsscore : float
            Polarity score for the input doc (not normalized)
        """
        mlsscore = 0
        # check for single words. A word could be a lemma or a derivation
        for word in doc.split():
            negated = False
            if "_NEG" in word:
                negated = True
                word = word.replace("_NEG", "")
            lem_word = self.lemmatize_spa(word)
            ############################################
            negated = negated and self.negation_handling
            ############################################
            if word in self.mlsent.keys():
                if negated:
                    # print("Word_NEG:", word)
                    mlsscore = mlsscore - self.mlsent[word]
                else:
                    # print("Word:", word)
                    mlsscore = mlsscore + self.mlsent[word]
            elif lem_word in self.mlsent.keys():
                if negated:
                    # print("Lemma_NEG:", lem_word)
                    mlsscore = mlsscore - self.mlsent[lem_word]
                else:
                    # print("Lemma:", lem_word)
                    mlsscore = mlsscore + self.mlsent[lem_word]
        if mlsscore > self.max_score:
            self.max_score = mlsscore
        if mlsscore < self.min_score:
            self.min_score = mlsscore
        tokens = doc.split()
        # check ngrams with 2, 3 and 4 words. ML Senticon has polarity scores
        # for multi-word phrases.
        if self.n_gram_handling:
            bigrams = ngrams(tokens, 2)
            trigrams = ngrams(tokens, 3)
            fourgrams = ngrams(tokens, 4)
            lngrams = [bigrams, trigrams, fourgrams]
            for phrases in lngrams:
                for phrase in phrases:
                    strphrase = ' '.join(phrase)
                    negated = False
                    if "_NEG" in strphrase:
                        negated = True
                        strphrase = strphrase.replace("_NEG", "")
                    strphrase = strphrase.lower()
                    ############################################
                    negated = negated and self.negation_handling
                    ############################################             
                    if strphrase in self.mlsent:
                        if negated:
                            # print("ngram_NEG:", strphrase)
                            mlsscore = mlsscore - self.mlsent[strphrase]
                        else:
                            # print("ngram:", strphrase)
                            mlsscore = mlsscore + self.mlsent[strphrase]
        return mlsscore


    def normalize_scores(self, results):
        """
        Normalize polarity scores into the range [-1,1] and 
        recalculates predicted sentiment label according to
        the normalized score.

        Notes: tests with normalization have not been conclusive
        enough. Normalized scores depend too heavily on the max and
        min scores obtained in a given document set. 
        Thus, a normalized score is not comparable with the normalized
        score from another document set, with its own max and min values.

        If scores are not normalized, they are are absolute values 
        and comparable with the scores obtained with other document
        set. This is the main reason why we opted for not using
        this method and not normalizing any score.
        """

        normalized = []
        max_val = self.max_score
        min_val = self.min_score
        limit = max([abs(max_val), abs(min_val)])
        #no need to normalize. All docs are neutral
        if max_val == 0 and min_val == 0:
            return results
        for (doc, sentiment, score) in results:
            n_score = score/limit
            if n_score < self.neu_inf_lim:
                n_sentiment = "neg"
            elif n_score < self.neu_sup_lim:
                n_sentiment = "neu"
            else:
                n_sentiment = "pos"
            normalized.append((doc, n_sentiment, n_score))
        return normalized

        
    def get_polarity_score(self, doc):
        """
        Returns the polarity score for a given doc.
        This score ranges from -1 to 1, were -1 is extreme negative
        and 1 means extreme positive.

        """

        if self.algorithm == "nltk_vader":
            return self._sia.polarity_scores(doc)["compound"]
        elif self.algorithm == "ML-Senticon":
            return self.spa_polarity_score(doc)

    def analyze_doc(self, doc):
        """
        Analyzes a given doc and returns a tuple
        (doc, predicted sentiment, polarity score)
        where doc is the original doc;
        predicted sentiment can be 'pos', 'neu' or 'neg'
        for positive, neutral and negative sentiment respectevely;
        and polarity score is a float that ranges from -1 to 1.
        """
        # pre processing stage
        pp_doc = clean_emojis(doc)
        if self.translate:
            pp_doc = translate_doc(doc, src=self.src_lang, dest="en")
        if self.language == "spanish":
            pp_doc = mark_negation_es(doc, join_words=True)
        else:
            pp_doc = tokenize_and_remove_stop_words(text=pp_doc, 
                                                    join_words=True)
        # get polarity score from pre processed doc
        score = self.get_polarity_score(pp_doc)
        # determine polarity from score and thresholds
        if score < self.neu_inf_lim:
            predicted_sentiment = "neg"
        elif score < self.neu_sup_lim:
            predicted_sentiment = "neu"
        else:
            predicted_sentiment = "pos"
        return (doc, predicted_sentiment, score)

    def analyze_docs(self, docs):
        """
        Analyzes a document collection by applying the analyze_doc() method
        for each document.
        All the results are stored in the _tagged_docs attribute.
        Normalize the results if needed.
        """
        results = []
        for doc in docs:
            results.append(self.analyze_doc(doc))
        if self.need_normalization and len(docs) > 1:
            results = self.normalize_scores(results)
        self._tagged_docs = results


    @property
    def tagged_docs(self):
        return self._tagged_docs
    


analyzer = SentimentAnalyzer()
analyzer.load_spa_resources()
analyzer.lemmatize_spa(word for word in text for text in tweet_texts)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:07:11 2017

@author: jorgesaldivar
"""

import nltk
import re
from cca_core.utils import tokenize_and_remove_stop_words, download_stop_words


class ConceptExtractor():
    ''' 
    
    Extract the most common concepts from a collection of documents.
    
    Parameters
    ----------
    num_concepts : int, 5 by default
        The number of concepts to extract.
    
    context_words : list, empty list by default
        List of context-specific words that should notbe considered in the 
        analysis.
    
    ngram_range: tuple, (1,1) by default
        The lower and upper boundary of the range of n-values for different 
        n-grams to be extracted. All values of n such that 
        min_n <= n <= max_n will be used.
    
    pos_vec: list, only words tagged as nouns (i.e., ['NN', 'NNP']) are 
    considered by default
        List of tags related with the part-of-speech that 
        should be considered in the analysis. Please check
        http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html 
        for a complete list of tags.
    
    consider_urls: boolean, False by default
        Whether URLs should be removed or not.
    
    language: string, 'english' by default
        Language of the documents. Only the languages supported by the
        library NLTK are supported.
    '''
    _reg_exp_urls = r'^https?:\/\/.*[\r\n]*'
    
    def __init__(self, num_concepts=5, context_words=[], 
                 ngram_range=(1,1), pos_vec=['NN', 'NNP'], 
                 consider_urls=False, language='english'):
        self.num_concepts = num_concepts
        self.context_words = context_words
        self.ngram_range = ngram_range
        self.pos_vec = pos_vec
        self.consider_urls = consider_urls
        self.language = language
        # properties
        self._docs = None
        self._number_words = 0
        self._unique_words = 0
        self._common_concepts = []
        # download stop words in case they weren't already downloaded
        download_stop_words()
    
    def extract_concepts(self, docs):
        '''
        Extract the most common concepts in the collection of 
        documents.
        
        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings
        
        Returns
        -------
        self : ConceptExtractor
        
        '''        
        self._docs = docs
        # tokenize documents
        tokenized_docs = []
        for doc in self._docs:
            if not self.consider_urls:
                doc = re.sub(self._reg_exp_urls, '', doc, 
                              flags=re.MULTILINE)
            tokenized_doc = tokenize_and_remove_stop_words(doc, 
                                                           self.context_words,
                                                           language=self.language)
            tokenized_docs.append(tokenized_doc)
            
        # consider only the part-of-speech (pos) required
        tagged_senteces = [nltk.pos_tag(token) for token in tokenized_docs]
        pos_tokens = [tagged_token for tagged_sentence in tagged_senteces 
                      for tagged_token in tagged_sentence if tagged_token[1] 
                      in self.pos_vec]
        tokens = [pos_token[0] for pos_token in pos_tokens]
        
        # compute most frequent words
        fdist = nltk.FreqDist(tokens)
        common_words = fdist.most_common(self.num_concepts)
        self._unique_words = fdist.keys()
        self._number_words = sum([i[1] for i in fdist.items()])
        
        # compute most frequent n-grams
        min_n, max_n = self.ngram_range
        common_bigrams = []
        common_trigrams = []
        if min_n == 1:
            if max_n == 1:
                self._common_concepts = common_words[:self.num_concepts]
                return self
            else:
                if max_n <= 3:
                    for i in range(min_n, max_n):
                        if i==1:
                            bgs = nltk.bigrams(tokens)
                            fdist = nltk.FreqDist(bgs)
                            common_bigrams = fdist.most_common(self.num_concepts)
                        else:
                            bgs = nltk.trigrams(tokens)
                            fdist = nltk.FreqDist(bgs)
                            common_trigrams = fdist.most_common(self.num_concepts)
                else:
                    raise Exception('The max number in the n-gram range \
                                    cannot be larger than 3')
        else:
            raise Exception('The minimun number in the n-gram range \
                            must be equal to 1')
        
        # make list of common concepts considering n-grams
        least_freq_common_word = common_words[-1][1]
        ngrams_to_consider = []
        # save relevant ngrams        
        for bigram in common_bigrams:
            if bigram[1] > least_freq_common_word:
                ngrams_to_consider.append(bigram)
            else:
                break
        for trigram in common_trigrams:
            if trigram[1] > least_freq_common_word:
                ngrams_to_consider.append(trigram)
            else:
                break
        # delete word of the ngrams from the list of common words to avoid 
        # duplicates
        idx_elements_to_remove = []
        for ngram in ngrams_to_consider:
            idx_elements_to_remove = [i for word in ngram[0] for i in 
                                      range(len(common_words)) 
                                      if word == common_words[i][0]]
        for idx in idx_elements_to_remove:
            del common_words[idx]
        # add to list of common words the relevant ngrams
        common_words.extend(
                [
                (' '.join(ngram[0]), ngram[1]) for ngram in ngrams_to_consider
                ]
        )
        # order list
        self._common_concepts = sorted(common_words, key=lambda tup: tup[1],
                                       reverse=True)
        # select the first n concepts
        self._common_concepts = self._common_concepts[:self.num_concepts]       
        return self
    
    @property
    def docs(self):
        return self._docs
    
    @property
    def total_words(self):
        return self._number_words
    
    @property
    def unique_words(self):
        return self._unique_words
    
    @property
    def common_concepts(self):
        return self._common_concepts

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:17:21 2017

@author: jorgesaldivar
"""

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from cca_core.utils import tokenize_and_remove_stop_words, tokenize_and_stem,\
                  download_stop_words
from cca_core.concept_extraction import ConceptExtractor


class DocumentClustering:
    '''
    Cluster documents by similarity using the k-means algorithm.
    
    Parameters
    ----------
    num_clusters : int, 5 by default
        The number of clusters in which the documents will be grouped.
    
    context_words : list, empty list by default
        List of context-specific words that should notbe considered in the 
        analysis.
        
    ngram_range: tuple, (1,1) by default
        The lower and upper boundary of the range of n-values for different 
        n-grams to be extracted. All values of n such that 
        min_n <= n <= max_n will be used.
    
    min_df: float in range [0.0, 1.0] or int, default=0.1
        The minimum number of documents that any term is contained in. It 
        can either be an integer which sets the number specifically, or a 
        decimal between 0 and 1 which is interpreted as a percentage of all 
        documents.
    
    max_df: float in range [0.0, 1.0] or int, default=0.9
        The maximum number of documents that any term is contained in. It 
        can either be an integer which sets the number specifically, or a 
        decimal between 0 and 1 which is interpreted as a percentage of all 
        documents.
    
    consider_urls: boolean, False by default
        Whether URLs should be removed or not.
    
    language: string, 'english' by default
        Language of the documents. Only the languages supported by the
        library NLTK are supported.

    algorithm: string, 'k-means' by default
        Clustering algorithm use to group documents
        Currently available: k-means and agglomerative (hierarchical)
    
    use_idf: boolean, False by default
        If true, it will use TF-IDF vectorization for feature extraction.
        If false it will use only TF.
    
    '''
    
    _reg_exp_urls = r'^https?:\/\/.*[\r\n]*'
    
    def __init__(self, num_clusters=5, context_words=[], ngram_range=(1,1), 
                 min_df=0.1, max_df=0.9, consider_urls=False, 
                 language='english', algorithm="k-means", 
                 use_idf=False):
        self.num_clusters = num_clusters
        self.context_words = context_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.consider_urls = consider_urls
        self.language = language
        self.use_idf = use_idf
        # properties
        self._docs = None
        self._corpus = pd.DataFrame()
        self._model = None
        self._tfidf_matrix = {}
        self._features = []
        self._feature_weights = {}
        self._num_docs_per_clusters = {}
        self._clusters = []
        self._algorithm = algorithm
        self._silhouette_score = 0
        self._calinski_harabaz_score = 0
        # download stop words in case they weren't already downloaded
        download_stop_words()
    
    def clustering(self, docs):
        '''
        Cluster, by similarity, a collection of documents into groups.
        
        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings
        
        Returns
        -------
        self : DocumentClustering
        '''
        
        self._docs = docs
        # clean and stem documents
        stemmed_docs = []
        for doc in self._docs:
            if not self.consider_urls:
                doc = re.sub(self._reg_exp_urls, '', doc, flags=re.MULTILINE)
            stemmed_docs.append(tokenize_and_stem(doc, self.context_words,
                                                  join_words=True,
                                                  language=self.language))
        # create td-idf matrix
        tfidf_vectorizer = TfidfVectorizer(max_df=self.max_df, 
                                            min_df=self.min_df,
                                            use_idf=self.use_idf,
                                            ngram_range=self.ngram_range)
        #fit the vectorizer to ideas
        try:
            self._tfidf_matrix = tfidf_vectorizer.fit_transform(stemmed_docs)
            self._features = tfidf_vectorizer.get_feature_names()
            weights = np.asarray(self._tfidf_matrix.mean(axis=0)).ravel().tolist()
            weights_df = pd.DataFrame({'term': self._features, 'weight': weights})
            self._feature_weights = weights_df.sort_values(by='weight', 
                                                          ascending=False). \
                                                          to_dict(orient='records')
        except ValueError as error:
            raise Exception(error)
        # compute clusters
        if self._algorithm == "agglomerative":
            self._model = AgglomerativeClustering(n_clusters=self.num_clusters)
            self._model.fit(self._tfidf_matrix.toarray())
        elif self._algorithm == "k-means":
            self._model = KMeans(n_clusters=self.num_clusters)
            self._model.fit(self._tfidf_matrix)
        self._clusters = self._model.labels_.tolist()
        # create a dictionary of the docs and their clusters
        docs_clusters = {'docs': self._docs, 'cluster': self._clusters}
        docs_clusters_df = pd.DataFrame(docs_clusters, index = [self._clusters] , 
                                        columns = ['doc', 'cluster'])
        # save the number of documents per cluster
        self._num_docs_per_clusters = dict(docs_clusters_df['cluster']. \
                                           value_counts())
        self._silhouette_score =  metrics.silhouette_score(self._tfidf_matrix,
                                                           self._model.labels_,
                                                            metric='euclidean')
        self._calinski_harabaz_score = metrics.calinski_harabaz_score(
                                                 self._tfidf_matrix.toarray(),
                                                 self._model.labels_)
        return self
    
    def top_terms_per_cluster(self, num_terms_per_cluster=3):
        '''
        Compute the top 'n' terms per cluster.
        
        Parameters
        ----------
        num_terms_per_cluster: int, default=3
            The number of terms per clusters that should be returned
        
        Returns
        -------
        top_terms : Dictionary of clusters and their top 'n' terms
        '''
        clusters_dic = {str(l): [] for l in set(self._clusters)}
        top_terms = {k:[] for k in clusters_dic.keys()}
        for i in range(0, len(self._clusters)):
            label = str(self._clusters[i])
            clusters_dic[label].append(self._docs[i])      
        for c,l in clusters_dic.items():
            ce = ConceptExtractor(num_concepts=num_terms_per_cluster,
                                  language=self.language, 
                                  context_words=self.context_words,
                                  pos_vec=['NN', 'NNP', 'NNS', 'NNPS'])
            ce.extract_concepts(l)
            top_terms[c] = ce.common_concepts
        return top_terms

    def get_coordinate_vectors(self):
        '''
        First, the function computes the cosine similarity of each document. 
        Cosine similarity is measured against the tf-idf matrix and can be used 
        to generate a measure of similarity between each document and the other 
        documents in the corpus.
        
        Then, it converts the dist matrix into a 2-dimensional array of 
        coordinate vectors.
        
        
        Returns
        -------
        coor_vecs: Dictionary that maps each document with its corresponding
        cluster and its x and y coordinate.
        '''
        
        MDS()
        dist = 1 - cosine_similarity(self._tfidf_matrix)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(dist) # shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]
        # create a dictionary that has the result of the MDS plus the cluster 
        # numbers and documents
        coor_vecs = dict(x=xs, y=ys, label=self._clusters, docs=self._docs)
        return coor_vecs
    
    @property
    def docs(self):
        return self._docs
    
    @property
    def features(self):
        return self._features
    
    @property
    def num_docs_per_cluster(self):
        return self._num_docs_per_clusters

class IterativeDocumentClustering:
    '''
    Cluster documents using the DocumentClustering class previously
    defined. If one of the clusters is too big, it clusters it again
    and repeat the process until all clusters are small enough.
    
    Parameters
    ----------
    num_clusters : int, 5 by default
        The number of clusters in which the documents will be grouped.
        If a given cluster is too big it will be re clustered so there
        could be more clusters than num_clusters.
    
    context_words : list, empty list by default
        List of context-specific words that should notbe considered in the 
        analysis.
        
    ngram_range: tuple, (1,1) by default
        The lower and upper boundary of the range of n-values for different 
        n-grams to be extracted. All values of n such that 
        min_n <= n <= max_n will be used.
    
    min_df: float in range [0.0, 1.0] or int, default=0.1
        The minimum number of documents that any term is contained in. It 
        can either be an integer which sets the number specifically, or a 
        decimal between 0 and 1 which is interpreted as a percentage of all 
        documents.
    
    max_df: float in range [0.0, 1.0] or int, default=0.9
        The maximum number of documents that any term is contained in. It 
        can either be an integer which sets the number specifically, or a 
        decimal between 0 and 1 which is interpreted as a percentage of all 
        documents.
    
    consider_urls: boolean, False by default
        Whether URLs should be removed or not.
    
    language: string, english by default
        Language of the documents. Only the languages supported by the
        library NLTK are supported.

    threshold: float, 0.9 by default
        Percentage of the docs that defines the maximun size for a cluster.abs

    n_sub_clusters: integer, 3 by default
        Number of sub cluster on which any big cluster will be re clustered.
    
    num_temrs: integer, 3 by default
        Number of top terms per cluster
    '''
    
    def __init__(self, num_clusters=5, context_words=[], ngram_range=(1,1), 
                min_df=0.05, max_df=0.9, consider_urls=False, 
                language='english', threshold=0.6, n_sub_clusters=3,
                num_terms=6, use_idf=False):
        self.num_clusters = num_clusters
        self.context_words = context_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.consider_urls = consider_urls
        self.language = language
        self.threshold = threshold
        self.n_sub_clusters = n_sub_clusters
        self.num_terms = num_terms
        self.use_idf = use_idf
        self._clusters_data = {}

    def cluster_subset(self, docs, coords=None, num_clusters=5):
        '''
        Cluster a set of docs into num_clusters groups

        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings
        
        coords: iterable
            An iterable which yields a list of tuple of (x,y) form where x
            and y represent bidimensional coordinates of each doc.
            If None, it uses the get_coordinate_vectors method of the
            DocumentClustering class to calculate new coordinates.

        num_clusters: int, 5 by default
            The number of clusters in which the documents will be grouped.

        Returns
        -------
        result: dictionary where keys are clusters labels and values are list
        of the form (t,x,y) where t is the text of a document, and x & y are
        the coordinates of the document.

        top_terms: dictionary where keys are clusters labels and values are
        strings that have the top termns per clusters.
        '''

        dc = DocumentClustering(num_clusters=num_clusters,
                                context_words=self.context_words,
                                ngram_range=self.ngram_range,
                                min_df=self.min_df,
                                max_df=self.max_df,
                                use_idf=self.use_idf)
        dc.clustering(docs)
        vec = dc.get_coordinate_vectors()
        if coords != None:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
        else:
            xs = vec["x"]
            ys = vec["y"]
        labels = vec["label"]
        texts = vec["docs"]
        result = {str(l): [] for l in set(labels)}
        for i in range(0, len(labels)):
            cluster = str(labels[i])
            data = (texts[i], xs[i], ys[i])
            result[cluster].append(data)

        top_terms = {str(c): tt for c,tt in dc.top_terms_per_cluster\
                                             (self.num_terms).items()}
        return result, top_terms

    def clustering(self, docs):
        '''
        Call cluster_subset method iteratively until all groups are small 
        enough.

        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings
        '''
        top_terms = {}
        limit =  int(self.threshold*len(docs))
        #first time cluster_subset is called with the num_clusters attribute
        result, top_terms = self.cluster_subset(docs=docs,
                                                num_clusters=self.num_clusters)
        while True:
            n_docs = {c: len(l) for c,l in result.items()}
            re_cluster = [c for c,n in n_docs.items() if n > limit]
            if len(re_cluster) == 0:
                break
            # re_cluster contains the labels of groups that are over the limit
            for rc in re_cluster:
                # remove big cluster from final result
                rc_data = result.pop(rc)
                rc_docs = [t for (t,x,y) in rc_data]
                saved_coords = [(x,y) for (t,x,y) in rc_data]
                # cluster_subset is called with the n_sub_clusters attribute
                # when re-clustering
                new_res, new_terms = self.cluster_subset(docs=rc_docs, 
                                             coords=saved_coords,
                                             num_clusters=self.n_sub_clusters)
                # add new clusters to final result
                for nc,l in new_res.items():
                    result[rc+"."+nc] = l
                # remove top terms of big clusters
                top_terms.pop(rc)
                # add new clusters' top terms
                for nc,tt in new_terms.items():
                    top_terms[rc+"."+nc] = tt
        self._clusters_data = result
        self._top_terms = top_terms
    
    @property
    def clusters_data(self):
        return self._clusters_data

    @property
    def top_terms_per_cluster(self):
        return self._top_terms
    
    
import collections
from nltk import NaiveBayesClassifier, DecisionTreeClassifier
from nltk.metrics import precision, recall, f_measure
from nltk.classify import apply_features, accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from cca_core.utils import clean_html_tags, shuffled, tokenize_and_stem
from cca_core.concept_extraction import ConceptExtractor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


class DocumentClassifier():
    '''
    Train a classifier with labeled documents and classify new documents 
    into one of the labeled clases.
    We call 'dev docs' to the documents set provided for training the 
    classifier. These 'dev docs' are splitted into two sub sets: 'train docs' 
    and 'test docs' that would be used to train and test the machine learning
    model respectively.

    Parameters
    ----------
        train_p : float, 0.8 by default
            The proportion of the 'dev docs' used as 'train docs'
            Use values greater than 0 and lower than 1.
            The remaining docs will be using as 'test docs'
    
    eq_label_num : boolean, True by default
        If true, 'train docs' will have equal number of documents for each
        class. This number will be the lowest label count.
    
    complete_p : boolean, True by default
        Used when eq_label_num is True, but the lowest label count is not
        enough for getting the train_p proportion of 'train docs'. If this 
        attribute is True, more documents from 'test docs' will be moved
        to 'train docs' until we get train_p

    n_folds : integer, 10 by default
        Number of folds to be used in k-fold cross validation technique for
        choosing different sets as 'train docs'

    vocab_size : integer, 500 by default
        This is the size of the vocabulary set that will be used for extracting
        features out of the docs

    t_classifier : string, 'NB' by default
        This is the type of classifier model used. Available types are 'NB' 
        (Naive Bayes), 'DT' (decision tree), 'RF' (Random Forest), and 'SVM'
        (Support Vector Machine)

    language: string, 'english' by default
        Language on which documents are written

    stem: boolean, False by deafault
        If True, stemming is applied to feature extraction

    train_method: string, 'all_class_train' by default
        Choose the method to train the classifier. There are two options:
        'all_class_train' and 'cross_validation'
    '''

    def __init__(self, train_p=0.8, 
                 eq_label_num=True,  
                 complete_p=True, 
                 n_folds=10,
                 vocab_size=250, 
                 t_classifier="NB", 
                 language="english", 
                 stem=False,
                 train_method="all_class_train"):
        self.train_p = train_p
        self.eq_label_num = eq_label_num
        self.complete_p = complete_p
        self.n_folds = n_folds
        self.vocab_size = vocab_size
        self.t_classifier = t_classifier
        self.language = language
        self.stem = stem
        self.train_method = train_method
        self._vocab = []
        self._classified_docs = []
        self._classifier = None
        self._accuracy = 0
        self._precision = {}
        self._recall = {}
        self._f_measure = {}
        self._train_docs = []
        self._test_docs = []

    def split_train_and_test(self, docs):
        '''
        Split the 'dev docs' set into the 'train docs' and 'test docs' subsets

        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings

        '''

        categories_count = self.count_categories(docs)
        label_limit = min([c for (k,c) in categories_count.items()])
        labeled_docs = {}
        train_docs = []
        test_docs = []
        # Split docs by label
        for (cat,count) in categories_count.items():
            labeled_docs[cat] = shuffled([t for (t,k) in docs if k == cat])
        if self.eq_label_num:
            # Select the same number of doc for all labels
            for cat, cat_docs in labeled_docs.items():
                cat_limit = label_limit
                cat_train_docs = cat_docs[:cat_limit]
                cat_test_docs = cat_docs[cat_limit:]
                train_docs += [(doc, cat) for doc in cat_train_docs]
                test_docs += [(doc, cat) for doc in cat_test_docs]
            l_train = len(train_docs)
            l_docs = len(docs)
            l_test = len(test_docs)
            actual_p = l_train / l_docs
            # If the training proportion is not 
            if self.complete_p == True and actual_p < self.train_p:
                shuffled_extra = shuffled(test_docs)
                extra_i = 0
                while(actual_p < self.train_p and extra_i < l_test):
                    aux_l_train = l_train + extra_i
                    actual_p = aux_l_train / l_docs
                    extra_i += 1
                train_docs += shuffled_extra[:extra_i]
                test_docs = shuffled_extra[extra_i:]
        else:
            label_limit = int(self.train_p * len(docs))
            shuffled_docs = shuffled(docs)
            train_docs = shuffled_docs[:label_limit]
            test_docs = shuffled_docs[label_limit:]
        self._train_docs = train_docs
        self._test_docs = test_docs

    def cross_validation_train(self, dev_docs):
        '''
        Applies k-fold cross validation technique to split the docs into different
        pairs of training and testing sets. For each pair, it trains and evals the
        a classifier, choosing the one with the best accuracy

        Parameters
        ----------
        dev_docs: iterable
            An iterable which yields a list of strings

        '''
        dev_docs = shuffled(dev_docs)
        accuracies = []
        best_accuracy = 0
        subset_size = int(len(dev_docs)/self.n_folds)

        for i in range(self.n_folds):
            classifier_list = []
            train_docs = (dev_docs[(i + 1) * subset_size:] + \
                          dev_docs[:i * subset_size])
            test_docs = dev_docs[i * subset_size:(i + 1) * subset_size]
            train_set = apply_features(self.get_doc_features, train_docs)
            if self.t_classifier == "NB":
                classifier = NaiveBayesClassifier.train(train_set)
            elif self.t_classifier == "DT":
                classifier = DecisionTreeClassifier.train(train_set)
            elif self.t_classifier == "RF":
                classifier = SklearnClassifier(RandomForestClassifier())\
                                                       .train(train_set)
            elif self.t_classifier == "SVM":
                classifier = SklearnClassifier(LinearSVC(), sparse=False)\
                                                         .train(train_set)

            classifier_list.append(classifier)
            test_set = apply_features(self.get_doc_features, test_docs, True)
            accuracies.append((accuracy(classifier, test_set)) * 100)

            if accuracies[-1] > best_accuracy:
                best_accuracy = accuracies[-1]
                self._classifier = classifier
                self._train_docs = train_docs
                self._test_docs = test_docs
    
    def all_class_train(self, dev_docs):
        '''
        Train classifier with train_p percentage of all classes. The remaining
        docs of each class is used for testing.

        Parameters
        ----------
        dev_docs: iterable
            An iterable which yields a list of strings
        '''
        categories_count = self.count_categories(dev_docs)
        
        labeled_docs = {}
        for (cat,count) in categories_count.items():
            labeled_docs[cat] = shuffled([t for (t,k) in dev_docs if k == cat])

        train_docs = []
        test_docs = []

        for cat, l in labeled_docs.items():
            cat_limit = int(self.train_p * len(l))
            train_docs += [(t, cat) for t in l[:cat_limit]]
            test_docs += [(t, cat) for t in l[cat_limit:]]

        self._train_docs = train_docs
        self._test_docs = test_docs

        train_set = apply_features(self.get_doc_features, self._train_docs)	
        # create and train the classification model according to t_classifier	
        if self.t_classifier == "NB":	
            self._classifier = NaiveBayesClassifier.train(train_set)	
        elif self.t_classifier == "DT":	
            self._classifier = DecisionTreeClassifier.train(train_set)	
        elif self.t_classifier == "RF":	
            self._classifier = SklearnClassifier(RandomForestClassifier())\
                                                         .train(train_set)	
        elif self.t_classifier == "SVM":	
            self._classifier = SklearnClassifier(LinearSVC(), sparse=False)\
                                                          .train(train_set)
    
    def count_categories(self, docs):
        '''
        Count how many documents of each class are in the 'dev docs' set
        
        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings

        Returns
        -------
        counters: dictionary
            A dictiionary where each item is the number of docs for a class
        '''

        categories = set([c for (t,c) in docs])
        counters = {}
        for cat in categories:
            counters[cat] = 0
        for (text, cat) in docs:
            counters[cat] += 1
        self._categories = sorted(categories)
        return counters

    def get_doc_features(self, doc):
        '''
        Extract features of a document, checking the presence of the words
        in the vocabulary

        Parameters
        ----------
        doc: string
            The doc from which features will be extracted

        Returns
        -------
        features: dictionary
            A dictionary where each item indicates the presence of a
            word from the vocabulary in the input doc
        '''

        features = {}
        for word in self._vocab:
            features['contains({})'.format(word)] = (word in doc)
        return features


    def train_classifier(self, dev_docs):
        '''
        Create the features vocabulary from 'dev docs', 
        Split 'dev docs', train the classifier with 'train docs',
        Evaluate accuracy with 'test docs'

        Parameters
        ----------
        dev_docs: iterable
            An iterable which yields a list of strings
        '''
        # create vocabulary for feature extraction
        ce = ConceptExtractor(num_concepts=self.vocab_size, 
                              language=self.language,
                              pos_vec=['NN', 'NNP', 'NNS', 'NNPS'])
        ce.extract_concepts([t for (t,c) in dev_docs])
        self._vocab = sorted([c for (c,f) in ce.common_concepts], key=str.lower)
        if (self.stem):
            self._vocab = [tokenize_and_stem(w, language=self.language)[0] \
                                                    for w in self._vocab]

        if self.train_method == "cross_validation":
            self.cross_validation_train(dev_docs)
        elif self.train_method == "all_class_train":
            self.all_class_train(dev_docs)


    def eval_classifier(self):
        '''
        Test the model and calculates the metrics of accuracy, precision,
        recall and f-measure
        '''
        test_set = apply_features(self.get_doc_features, self._test_docs, True)
        self._accuracy = accuracy(self._classifier, test_set)
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        
        for i, (feats, label) in enumerate(test_set):
            refsets[label].add(i)
            observed = self._classifier.classify(feats)
            testsets[observed].add(i)
        self.count_categories(self._train_docs)
        for cat in self._categories:
            self._precision[cat] = precision(refsets[cat], testsets[cat])
            self._recall[cat] = recall(refsets[cat], testsets[cat])
            self._f_measure[cat] = f_measure(refsets[cat], testsets[cat])


    def classify_docs(self, docs):
        '''
        First train the classifier with the labeled data.
        Then classifies the unlabeled data.

        Parameters
        ----------
        docs: iterable
            An iterable which yields a list of strings
        '''

        dev_docs = [(t, c) for (t, c) in docs if c!=""]
        unlabeled_docs = [t for (t, c) in docs if c==""]
        self.train_classifier(dev_docs)
        self.eval_classifier()
        results = []
        for doc in unlabeled_docs:
            doc_feats = self.get_doc_features(doc)
            result = self._classifier.classify(doc_feats)
            results.append((doc, result))
        self._classified_docs = results
        self._final_cat_count = self.count_categories(dev_docs+results)
    
    @property
    def classified_docs(self):
        return self._classified_docs

    @property    
    def accuracy(self):
        return self._accuracy
    
    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def f_measure(self):
        return self._f_measure

    @property
    def category_count(self):
        return self._final_cat_count


