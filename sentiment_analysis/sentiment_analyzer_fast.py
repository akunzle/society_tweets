
import csv
import os

# set the path to the folder containing CSV files
folder_path = "/Users/andreakunzle/society_tweets/tweets"

# get all the file names in the folder
file_names = os.listdir(folder_path)

# filter out non-CSV files
csv_file_names = [file for file in file_names if file.endswith('.csv')]

# print the list of CSV file names
print(csv_file_names)
print(len(csv_file_names))

def retrieve_csv_data(file_names):
    for csv_file_name in file_names:
        input_file = f'/Users/andreakunzle/society_tweets/tweets/{csv_file_name}'
        output_file = f'/Users/andreakunzle/society_tweets/sentiment_analysis/{csv_file_name}'
        with open(input_file, 'r', newline='', encoding='utf-8') as file, \
            open(output_file, 'w', newline='', encoding='utf-8') as out_file:
            reader = csv.DictReader(file)
            writer = csv.writer(out_file)
            writer.writerow(['postDate', 'responding'])  # Write the new header row
            for row in reader:
                if row['responding'].strip() and not row['responding'].startswith('Replying to'):
                    writer.writerow([row['postDate'], row['responding'].replace('\n', '')])
        with open(output_file, 'r', newline='', encoding='utf-8') as new_file: 
            reader = csv.reader(new_file)
            # Skip the header row
            next(reader)
            # Iterate through each row and save the 'responding' text
            tweet_texts = []
            post_dates = []
            for row in reader:
                tweet_texts.append(row[1])
                post_dates.append(row[0])
            print(f'{csv_file_name}_tweet_texts:', tweet_texts)
            print(f'{csv_file_name}_post_dates:', post_dates)

retrieve_csv_data(csv_file_names)

def read_tweet_texts(csv_file_names, root_path):
    for csv_file_name in csv_file_names:
        var_name = csv_file_name[:-4] + '_texts'
        print(var_name)
        file_path = os.path.join(root_path, csv_file_name)
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # skip header row
            tweet_texts = []
            for row in reader:
                tweet_texts.append(row[1])
        globals()[var_name] = tweet_texts

read_tweet_texts(csv_file_names, '/Users/andreakunzle/society_tweets/sentiment_analysis')
print(algoritmo_tweets_texts)
automatizacion_tweets_texts = automatizacion_tweets_texts
ciber_tweets_texts = ciber_tweets_texts
codificacion_tweets_texts = codificacion_tweets_texts
conacyt_tweets_texts = conacyt_tweets_texts
conectividad_tweets_texts = conectividad_tweets_texts
cripto_tweets_texts = cripto_tweets_texts
digital_tweets_texts = digital_tweets_texts
fibra_optica_tweets_texts = fibra_optica_tweets_texts
innovacion_tweets_texts = innovacion_tweets_texts
inteligencia_artificial_tweets_texts = inteligencia_artificial_tweets_texts
internet_tweets_texts = internet_tweets_texts
mitic_tweets_texts = mitic_tweets_texts
realidad_aumentada_tweets_texts = realidad_aumentada_tweets_texts
robot_tweets_texts = robot_tweets_texts
tecnologia_tweets_texts = tecnologia_tweets_texts
virtual_tweets_texts = virtual_tweets_texts

#Start cleaning the tweets

import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load the NLTK stop words in Spanish
stop_words_es = set(stopwords.words('spanish'))

def clean_tweets_in_vars(variable_names):
    cleaned_tweets = []
    for var_name in variable_names:
        tweet_texts = globals()[var_name]
        cleaned_var_name = var_name.replace('_tweets_texts', '') # remove '_tweet_texts' from the variable name
        cleaned_tweet_texts = [clean_tweet(tweet) for tweet in tweet_texts]
        globals()[f"{cleaned_var_name}_cleaned_tweets"] = cleaned_tweet_texts # create a new variable with the cleaned tweets
        cleaned_tweets.extend(cleaned_tweet_texts)
    return cleaned_tweets

def clean_tweet(tweet):
    # Remove URLs and web links
    tweet = re.sub(r'http\S+', '', tweet)

    # Add space to tweets that require it
    tweet = re.sub(r'\.(?=[^\s\d])', '. ', tweet)

    # Remove mentions
    tweet = re.sub(r'@\S+', '', tweet)

    # Remove numbers
    tweet = re.sub(r'[0-9]+', '', tweet)

    # Convert abbreviations 
    tweet = re.sub(r'Tqm', 'te quiero mucho', tweet)
    tweet = re.sub(r'Tqm', 'te quiero mucho', tweet)

    # Remove punctuation and special characters (except for '#')
    tweet = tweet.translate(str.maketrans('', '', string.punctuation.replace('#', '')+ '¿'))

    # Remove extra white spaces
    tweet = re.sub(r'\s+', ' ', tweet)

    return tweet

variable_names = ['automatizacion_tweets_texts', 'ciber_tweets_texts', 'codificacion_tweets_texts', 'conacyt_tweets_texts', 'conectividad_tweets_texts', 'cripto_tweets_texts', 'digital_tweets_texts', 'fibra_optica_tweets_texts', 'innovacion_tweets_texts', 'inteligencia_artificial_tweets_texts', 'internet_tweets_texts', 'mitic_tweets_texts', 'realidad_aumentada_tweets_texts', 'robot_tweets_texts', 'tecnologia_tweets_texts', 'virtual_tweets_texts']

algoritmo_var_names = ['algoritmo_tweets_texts']
cleaned_tweets = clean_tweets_in_vars(variable_names)
print(cleaned_tweets)
algoritmo_cleaned_tweets = clean_tweets_in_vars(algoritmo_var_names)
print(algoritmo_cleaned_tweets)

algoritmo_cleaned_tweets = algoritmo_cleaned_tweets
print(algoritmo_cleaned_tweets)
automatizacion_cleaned_tweets = automatizacion_cleaned_tweets
ciber_cleaned_tweets = ciber_cleaned_tweets
codificacion_cleaned_tweets = codificacion_cleaned_tweets
conacyt_cleaned_tweets = conacyt_cleaned_tweets
conectividad_cleaned_tweets = conectividad_cleaned_tweets
cripto_cleaned_tweets = cripto_cleaned_tweets
digital_cleaned_tweets = digital_cleaned_tweets
fibra_optica_cleaned_tweets = fibra_optica_cleaned_tweets
innovacion_cleaned_tweets = innovacion_cleaned_tweets
inteligencia_artificial_cleaned_tweets = inteligencia_artificial_cleaned_tweets
internet_cleaned_tweets = internet_cleaned_tweets
mitic_cleaned_tweets = mitic_cleaned_tweets
realidad_aumentada_cleaned_tweets = realidad_aumentada_cleaned_tweets
robot_cleaned_tweets = robot_cleaned_tweets
tecnologia_cleaned_tweets = tecnologia_cleaned_tweets
virtual_cleaned_tweets = virtual_cleaned_tweets


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
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
from cca_core.utils import tokenize_and_remove_stop_words, tokenize_and_stem, \
                  clean_emojis, translate_doc, mark_negation_es

for i in tqdm(range(10), mininterval=1):
    time.sleep(1)

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
    

analyzer = SentimentAnalyzer(language='spanish')
algoritmo_var_names_cleaned = ['algoritmo_cleaned_tweets']
cleaned_variables = ['automatizacion_cleaned_tweets', 'ciber_cleaned_tweets', 'codificacion_cleaned_tweets', 'conacyt_cleaned_tweets', 'conectividad_cleaned_tweets', 'cripto_cleaned_tweets', 'digital_cleaned_tweets', 'fibra_optica_cleaned_tweets', 'innovacion_cleaned_tweets', 'inteligencia_artificial_cleaned_tweets', 'internet_cleaned_tweets', 'mitic_cleaned_tweets', 'realidad_aumentada_cleaned_tweets', 'robot_cleaned_tweets', 'tecnologia_cleaned_tweets', 'virtual_cleaned_tweets']
def analyze_docs_in_var(variable_names):
    analyzed_tweets = []
    for var_name in tqdm(variable_names):
        tweet_texts = globals()[var_name]
        cleaned_var_name = var_name.replace('_cleaned_tweets', '') # remove '_tweet_texts' from the variable name
        analyze_tweet_texts = [analyzer.analyze_doc(tweet) for tweet in tweet_texts]
        globals()[f"{cleaned_var_name}_analyzed_tweets"] = analyze_tweet_texts # create a new variable with the cleaned tweets
        analyzed_tweets.extend(analyze_tweet_texts)
    return analyzed_tweets

analyze_docs_in_var(cleaned_variables)
analyze_docs_in_var(algoritmo_var_names_cleaned)

algoritmo_analyzed_tweets = algoritmo_analyzed_tweets

automatizacion_analyzed_tweets = automatizacion_analyzed_tweets
ciber_analyzed_tweets = ciber_analyzed_tweets
codificacion_analyzed_tweets = codificacion_analyzed_tweets
conacyt_analyzed_tweets = conacyt_analyzed_tweets
conectividad_analyzed_tweets = conectividad_analyzed_tweets
cripto_analyzed_tweets = cripto_analyzed_tweets
digital_analyzed_tweets = digital_analyzed_tweets
fibra_optica_analyzed_tweets = fibra_optica_analyzed_tweets
innovacion_analyzed_tweets = innovacion_analyzed_tweets
inteligencia_artificial_analyzed_tweets = inteligencia_artificial_analyzed_tweets
internet_analyzed_tweets = internet_analyzed_tweets
mitic_analyzed_tweets = mitic_analyzed_tweets
realidad_aumentada_analyzed_tweets = realidad_aumentada_analyzed_tweets
robot_analyzed_tweets = robot_analyzed_tweets
tecnologia_analyzed_tweets = tecnologia_analyzed_tweets
virtual_analyzed_tweets = virtual_analyzed_tweets


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:07:11 2017

@author: jorgesaldivar
"""

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


ce = ConceptExtractor(num_concepts=10, language='spanish')
concepts = ce.extract_concepts(cleaned_tweets)
print(concepts)

docs = ce.docs
print(docs)

concepts_common = ce.common_concepts
print(concepts_common)

def extract_concepts_in_var(variable_names):
    for var_name in tqdm(variable_names):
        tweet_texts = globals()[var_name]
        cleaned_var_name = var_name.replace('_cleaned_tweets', '') # remove '_tweet_texts' from the variable name
        extract_concepts_of_tweets = ce.extract_concepts(tweet_texts)
        var_docs = ce.docs
        var_common_concepts = ce.common_concepts
        globals()[f"{cleaned_var_name}_extracted_concepts"] = extract_concepts_of_tweets # create a new variable with the cleaned tweets
    return f"{cleaned_var_name}_extracted_concepts", var_docs, var_common_concepts

extract_concepts_in_var(cleaned_variables)
extract_concepts_in_var(algoritmo_var_names_cleaned)

algoritmo_extracted_concepts = algoritmo_extracted_concepts.common_concepts
print(algoritmo_extracted_concepts)
automatizacion_extracted_concepts = automatizacion_extracted_concepts.common_concepts
ciber_extracted_concepts = ciber_extracted_concepts.common_concepts
codificacion_extracted_concepts = codificacion_extracted_concepts.common_concepts
conacyt_extracted_concepts = conacyt_extracted_concepts.common_concepts
conectividad_extracted_concepts = conectividad_extracted_concepts.common_concepts
cripto_extracted_concepts = cripto_extracted_concepts.common_concepts
digital_extracted_concepts = digital_extracted_concepts.common_concepts
fibra_optica_extracted_concepts = fibra_optica_extracted_concepts.common_concepts
innovacion_extracted_concepts = innovacion_extracted_concepts.common_concepts
inteligencia_artificial_extracted_concepts = inteligencia_artificial_extracted_concepts.common_concepts
internet_extracted_concepts = internet_extracted_concepts.common_concepts
mitic_extracted_concepts = mitic_extracted_concepts.common_concepts
realidad_aumentada_extracted_concepts = realidad_aumentada_extracted_concepts.common_concepts
robot_extracted_concepts = robot_extracted_concepts.common_concepts
tecnologia_extracted_concepts = tecnologia_extracted_concepts.common_concepts
virtual_extracted_concepts = virtual_extracted_concepts.common_concepts

print(automatizacion_extracted_concepts)


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
        self._calinski_harabasz_score = 0
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
            self._features = tfidf_vectorizer.get_feature_names_out()
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
        self._calinski_harabasz_score = metrics.calinski_harabasz_score(
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

ctg = DocumentClustering(num_clusters=5, min_df=0.05, max_df=0.7, language='spanish')
clusters = ctg.clustering(cleaned_tweets)
top_clusters = ctg.top_terms_per_cluster()
coordinates = ctg.get_coordinate_vectors()
print(clusters)
print(top_clusters)
print(coordinates)

clusters_docs = ctg.docs
features = ctg.features
num_docs_p_cluster = ctg.num_docs_per_cluster
print(clusters_docs)
print(features)
print(num_docs_p_cluster)

def doc_clustering_in_var(variable_names):
    for var_name in tqdm(variable_names):
        tweet_texts = globals()[var_name]
        cleaned_var_name = var_name.replace('_cleaned_tweets', '') # remove '_tweet_texts' from the variable name
        clustering_of_tweets = ctg.clustering(tweet_texts)
        clusters_features_tweets = ctg.features
        clusters_num_docs_p_tweet_cluster = ctg.num_docs_per_cluster
        globals()[f"{cleaned_var_name}_clustering_tweets"] = clustering_of_tweets # create a new variable with the cleaned tweets
        globals()[f"{cleaned_var_name}_clustering_tweets_features"] = clusters_features_tweets # create a new variable with the cleaned tweets
        globals()[f"{cleaned_var_name}_clustering_tweets_num_docs_p_cluster"] = clusters_num_docs_p_tweet_cluster # create a new variable with the cleaned tweets
    return f"{cleaned_var_name}_clustering_tweets", f"{cleaned_var_name}_clustering_tweets_features", f"{cleaned_var_name}_clustering_tweets_num_docs_p_cluster"

doc_clustering_in_var(cleaned_variables)
doc_clustering_in_var(algoritmo_var_names_cleaned)

algoritmo_clustering_tweets = algoritmo_clustering_tweets.features

automatizacion_clustering_tweets = automatizacion_clustering_tweets.features
ciber_clustering_tweets = ciber_clustering_tweets.features
codificacion_clustering_tweets = codificacion_clustering_tweets.features
conacyt_clustering_tweets = conacyt_clustering_tweets.features
conectividad_clustering_tweets = conectividad_clustering_tweets.features
cripto_clustering_tweets = cripto_clustering_tweets.features
digital_clustering_tweets = digital_clustering_tweets.features
fibra_optica_clustering_tweets = fibra_optica_clustering_tweets.features
innovacion_clustering_tweets = innovacion_clustering_tweets.features
inteligencia_artificial_clustering_tweets = inteligencia_artificial_clustering_tweets.features
internet_clustering_tweets = internet_clustering_tweets.features
mitic_clustering_tweets = mitic_clustering_tweets.features
realidad_aumentada_clustering_tweets = realidad_aumentada_clustering_tweets.features
robot_clustering_tweets = robot_clustering_tweets.features
tecnologia_clustering_tweets = tecnologia_clustering_tweets.features
virtual_clustering_tweets = virtual_clustering_tweets.features


algoritmo_clustering_tweets_num = algoritmo_clustering_tweets.num_docs_per_cluster
automatizacion_clustering_tweets_num = automatizacion_clustering_tweets.num_docs_per_cluster
ciber_clustering_tweets_num = ciber_clustering_tweets.num_docs_per_cluster
codificacion_clustering_tweets_num = codificacion_clustering_tweets.num_docs_per_cluster
conacyt_clustering_tweets_num = conacyt_clustering_tweets.num_docs_per_cluster
conectividad_clustering_tweets_num = conectividad_clustering_tweets.num_docs_per_cluster
cripto_clustering_tweets_num = cripto_clustering_tweets.num_docs_per_cluster
digital_clustering_tweets_num = digital_clustering_tweets.num_docs_per_cluster
fibra_optica_clustering_tweets_num = fibra_optica_clustering_tweets.num_docs_per_cluster
innovacion_clustering_tweets_num = innovacion_clustering_tweets.num_docs_per_cluster
inteligencia_artificial_clustering_tweets_num = inteligencia_artificial_clustering_tweets.num_docs_per_cluster
internet_clustering_tweets_num = internet_clustering_tweets.num_docs_per_cluster
mitic_clustering_tweets_num = mitic_clustering_tweets.num_docs_per_cluster
realidad_aumentada_clustering_tweets_num = realidad_aumentada_clustering_tweets.num_docs_per_cluster
robot_clustering_tweets_num = robot_clustering_tweets.num_docs_per_cluster
tecnologia_clustering_tweets_num = tecnologia_clustering_tweets.num_docs_per_cluster
virtual_clustering_tweets_num = virtual_clustering_tweets.num_docs_per_cluster


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
    
icd = IterativeDocumentClustering(language='spanish')
print(icd)
subset = icd.cluster_subset(cleaned_tweets)
print(subset)
clust = icd.clustering(cleaned_tweets)
print(clust)

data_clust = icd.clusters_data
top_terms = icd.top_terms_per_cluster

print(data_clust)
print(top_terms)

def iterative_clustering_in_var(variable_names):
    for var_name in tqdm(variable_names):
        tweet_texts = (globals()[var_name])
        stop_words_removed = [tokenize_and_remove_stop_words(tweet, specific_words_to_delete=['como', 'de', 'cómo', 'tipo'], join_words=True, language='spanish') for tweet in automatizacion_cleaned_tweets]
        cleaned_var_name = var_name.replace('_cleaned_tweets', '') # remove '_tweet_texts' from the variable name
        iterative_tweets_subset = icd.cluster_subset(stop_words_removed)
        iterative_clusters_tweets = icd.clustering(stop_words_removed)
        data_iterative_clust = icd.clusters_data
        top_iterative_terms = icd.top_terms_per_cluster
        globals()[f"{cleaned_var_name}_it_clust_tweets"] = iterative_clusters_tweets # create a new variable with the cleaned tweets
        globals()[f"{cleaned_var_name}_it_clust_data"] = data_iterative_clust
        globals()[f"{cleaned_var_name}_it_clust_top_terms"] = top_iterative_terms
    return iterative_tweets_subset, f"{cleaned_var_name}_it_clust_tweets", f"{cleaned_var_name}_it_clust_data", f"{cleaned_var_name}_it_clust_top_terms"

iterative_clustering_in_var(cleaned_variables)
iterative_clustering_in_var(algoritmo_var_names_cleaned)

import json
from datetime import datetime

var_names = ['automatizacion_analyzed_tweets', 'ciber_analyzed_tweets', 'codificacion_analyzed_tweets', 'conacyt_analyzed_tweets', 'conectividad_analyzed_tweets', 'cripto_analyzed_tweets', 'digital_analyzed_tweets', 'fibra_optica_analyzed_tweets', 'innovacion_analyzed_tweets', 'inteligencia_artificial_analyzed_tweets', 'internet_analyzed_tweets', 'mitic_analyzed_tweets', 'realidad_aumentada_analyzed_tweets', 'robot_analyzed_tweets', 'tecnologia_analyzed_tweets', 'virtual_analyzed_tweets']

def sentiment_analysis_report(var_names):
    for var_name in var_names:
        cleaned_var_name = var_name.replace('_analyzed_tweets', '') # remove '_tweet_texts' from the variable name
        input_file = f'/Users/andreakunzle/society_tweets/sentiment_analysis/{cleaned_var_name}_tweets.csv'
        output_file = f'/Users/andreakunzle/society_tweets/sentiment_analysis/{cleaned_var_name}_tweets_analysis.csv'
        cleaned_tweets = globals()[f'{cleaned_var_name}_cleaned_tweets']
        with open(input_file, 'r', newline='', encoding='utf-8') as data_file: 
            reader = csv.reader(data_file)
            # Skip the header row
            next(reader)
            # Iterate through each row and save the dates
            tweet_dates = []
            for row in reader:
                tweet_dates.append(row[0])
        with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['postDates', 'tweets', 'sentiment_label', 'polarity_score'])  # Write the new header row
            for i in range(len(tweet_dates)): 
                writer.writerow([tweet_dates[i], cleaned_tweets[i], globals()[var_name][i][1], globals()[var_name][i][2]])

sentiment_analysis_report(var_names)

algoritmo_var_names = ['algoritmo_analyzed_tweets']
sentiment_analysis_report(algoritmo_var_names)


from numpyencoder import NumpyEncoder
from collections import OrderedDict

def tweets_stats_report(var_names):
    for var_name in var_names:
        cleaned_var_name = var_name.replace('_analyzed_tweets', '') # remove '_tweet_texts' from the variable name
        output_file = f'/Users/andreakunzle/society_tweets/sentiment_analysis/{cleaned_var_name}_tweets_analysis.csv'
        common_concepts = globals()[f'{cleaned_var_name}_extracted_concepts']
        concepts_dict = {k.capitalize(): v for k, v in common_concepts}
        clustering_features = globals()[f'{cleaned_var_name}_clustering_tweets_features'].tolist()
        num_docs_p_cluster = np.array((globals()[f'{cleaned_var_name}_clustering_tweets_num_docs_p_cluster']))
        it_clust_data = np.array((globals()[f'{cleaned_var_name}_it_clust_data']))
        it_clust_top_terms = np.array((globals()[f'{cleaned_var_name}_it_clust_top_terms']))
        with open(output_file, 'r', newline='', encoding='utf-8') as data_file: 
            reader = csv.reader(data_file)
            # Skip the header row
            next(reader)
            # Iterate through each row and save the dates
            tweet_dates = []
            for row in reader:
                tweet_dates.append(row[0])

        # Calculate post dates stats
        post_dates = [datetime.strptime(tweet, '%Y-%m-%dT%H:%M:%S.%fZ') for tweet in tweet_dates]
        num_tweets = len(post_dates)
        num_days = (max(post_dates) - min(post_dates)).days + 1
        num_weeks = num_days // 7
        num_months = (max(post_dates).year - min(post_dates).year) * 12 + (max(post_dates).month - min(post_dates).month) + 1
        num_years = max(post_dates).year - min(post_dates).year + 1
        avg_tweets_per_day = num_tweets / num_days
        avg_tweets_per_week = num_tweets / num_weeks
        avg_tweets_per_month = num_tweets / num_months
        avg_tweets_per_year = num_tweets / num_years
        peak_tweet_day = max(set(post_dates), key=post_dates.count).strftime('%Y-%m-%d')
        peak_tweet_hour = max(set(post_dates), key=post_dates.count).strftime('%H:00')
        time_between_tweets = [(post_dates[i+1] - post_dates[i]).total_seconds() for i in range(len(post_dates)-1)]
        avg_time_between_tweets = sum(time_between_tweets) / len(time_between_tweets)
        min_time_between_tweets = min(time_between_tweets)
        max_time_between_tweets = max(time_between_tweets)

        # Calculate sentiment analysis stats
        positive_tweets = len([tweet for tweet in globals()[var_name] if tweet[1] == 'pos'])
        neutral_tweets = len([tweet for tweet in globals()[var_name] if tweet[1] == 'neu'])
        negative_tweets = len([tweet for tweet in globals()[var_name] if tweet[1] == 'neg'])
        if positive_tweets > 0:
            pos_mean_polarity = sum([scores[2] for scores in globals()[var_name] if scores[1] == 'pos']) / positive_tweets
        else:
            pos_mean_polarity = 0
        if neutral_tweets > 0:
            neu_mean_polarity = sum([scores[2] for scores in globals()[var_name] if scores[1] == 'neu']) / neutral_tweets
        else:
            neu_mean_polarity = 0
        if negative_tweets > 0:
            neg_mean_polarity = sum([scores[2] for scores in globals()[var_name] if scores[1] == 'neg']) / negative_tweets
        else:
            neg_mean_polarity = 0
        
        # Define the OrderedDict with the desired key order
        stats_dict = OrderedDict([
            ('Post Dates Stats', OrderedDict([
                ('Average tweets per day', avg_tweets_per_day),
                ('Average tweets per week', avg_tweets_per_week),
                ('Average tweets per month', avg_tweets_per_month),
                ('Average tweets per year', avg_tweets_per_year),
                ('Peak tweet day', peak_tweet_day),
                ('Peak tweet hour', peak_tweet_hour),
                ('Average time between tweets', f"{int(avg_time_between_tweets // 86400)}d {int(avg_time_between_tweets // 3600)}h {int((avg_time_between_tweets % 3600) // 60)}m"),
                ('Minimum time between tweets', f"{int(min_time_between_tweets // 86400)}d {int(min_time_between_tweets // 3600)}h {int((min_time_between_tweets % 3600) // 60)}m {int(min_time_between_tweets % 60)}s"),
                ('Maximum time between tweets', f"{int(max_time_between_tweets // 86400)}d {int(max_time_between_tweets // 3600)}h {int((max_time_between_tweets % 3600) // 60)}m {int(max_time_between_tweets % 60)}s")
                ])),
            ('Number of tweets', num_tweets),
            ('Sentiment Analysis Stats', OrderedDict([
                ('Number of Positive Tweets', positive_tweets),
                ('Number of Neutral Tweets', neutral_tweets),
                ('Number of Negative Tweets', negative_tweets)
                ])),
            ('Polarity Score Stats', OrderedDict([
                ('Average Positive Score', pos_mean_polarity),
                ('Average Neutral Score', neu_mean_polarity),
                ('Average Negative Score', neg_mean_polarity)
                ])),
            ('Clustering of Tweets', OrderedDict([
                ('Cluster Features', clustering_features),
                ('Number of Docs per Cluster', num_docs_p_cluster)
                ])), 
            ('Common Concepts of Tweets and Counts', concepts_dict),
            ('Iterative Clustering of Tweets', OrderedDict([
                ('Clustering Tweets Data', it_clust_data),
                ('Top Terms per Cluster', it_clust_top_terms)
                ]))
        ])
        # Write to the JSON file
        with open(f'{cleaned_var_name}_tweets_stats_report.json', 'w') as f:
            json.dump(stats_dict, f, indent=4, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)

algoritmo_var_names = ['algoritmo_analyzed_tweets']

tweets_stats_report(algoritmo_var_names)















































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




trained_labels = ["innovativeness", "innovativeness", "innovativeness", "discomfort", "discomfort", "insecurity", "discomfort", "insecurity", "insecurity", "innovativeness", "optimism", "insecurity", "optimism", "insecurity", "insecurity", "optimism", "insecurity", "insecurity", "insecurity", "insecurity", "discomfort", "innovativeness", "discomfort", "optimism", "optimism", "innovativeness", "optimism", "discomfort", "insecurity", "discomfort", "insecurity", "innovativeness", "discomfort", "insecurity", "discomfort", "discomfort", "discomfort", "innovativeness", "optimism", "insecurity", "innovativeness", "innovativeness", "insecurity", "discomfort", "discomfort", "discomfort", "insecurity", "insecurity", "insecurity", "insecurity", "innovativeness", "discomfort", "optimism", "innovativeness", "insecurity", "optimism", "insecurity", "insecurity", "discomfort", "optimism", "discomfort", "discomfort", "discomfort", "insecurity", "insecurity", "optimism", "insecurity", "discomfort", "discomfort", "optimism", "insecurity", "insecurity", "insecurity", "discomfort", "insecurity", "discomfort", "optimism", "optimism", "innovativeness", "innovativeness", "insecurity", "insecurity", "insecurity", "insecurity", "discomfort", "optimism", "innovativeness", "insecurity", "insecurity", "optimism", "insecurity", "discomfort", "optimism", "discomfort", "innovativeness", "discomfort", "optimism", "innovativeness", "optimism", "optimism", "optimism", "optimism"]
print(len(trained_labels))

print(len(cleaned_tweets))

# Create a dictionary with the two lists
data = {'responding': cleaned_tweets, 'categories': trained_labels}

# Open a CSV file and write the dictionary to it
with open('algoritmo_trained_set.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['responding', 'categories'])
    writer.writeheader()
    for i in range(len(cleaned_tweets)):
        writer.writerow({'responding': cleaned_tweets[i], 'categories': trained_labels[i]})

import random

with open('algoritmo_trained_set.csv', 'r') as file:
    reader = csv.reader(file)
    
    # skip header row
    next(reader)
    
    # create formatted list of tuples
    formatted_list = [(row[0], row[1]) for row in reader]
    
    # calculate number of items to be saved in train_set
    train_size = int(0.8 * len(formatted_list))
    
    # randomly select items for train_set
    traini_set = random.sample(formatted_list, train_size)
    
    # replace the y value with "" for test_set
    testi_set = [(x, "") for x, y in formatted_list if (x, y) not in traini_set]
    
    # print train_set and test_set
    print("Train Set:", traini_set)
    print("Test Set:", testi_set)
    print(formatted_list)

input_set = traini_set + testi_set

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="NB",
                                language="spanish", stem=False, train_method="all_class_train")

# accuracy: 0.2222222222222222
# precision: {'discomfort': 0.0, 'innovativeness': 0.5, 'insecurity': 0.6, 'optimism': 0.0}, 
# num of categories{'insecurity': 38, 'optimism': 17, 'discomfort': 30, 'innovativeness': 17}
# recacll: {'discomfort': 0.0, 'innovativeness': 0.3333333333333333, 'insecurity': 0.5, 'optimism': 0.0}
# f-measure: {'discomfort': 0, 'innovativeness': 0.4, 'insecurity': 0.5454545454545454, 'optimism': 0}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="DT",
                                language="spanish", stem=False, train_method="all_class_train")

# accuracy: 0.2222222222222222
# precision: {'discomfort': 0.5, 'innovativeness': 0.16666666666666666, 'insecurity': 0.2222222222222222, 'optimism': 0.0}
# num of categories: {'insecurity': 40, 'optimism': 16, 'discomfort': 23, 'innovativeness': 23}
# recall: {'discomfort': 0.2, 'innovativeness': 0.3333333333333333, 'insecurity': 0.3333333333333333, 'optimism': 0.0}
# f-measure: {'discomfort': 0.2857142857142857, 'innovativeness': 0.2222222222222222, 'insecurity': 0.26666666666666666, 'optimism': 0}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="RF",
                                language="spanish", stem=False, train_method="all_class_train")

# accuracy: 0.2222222222222222
# precision: {'discomfort': 0.25, 'innovativeness': 0.0, 'insecurity': 0.2857142857142857, 'optimism': None}
# num of categories: {'insecurity': 35, 'optimism': 17, 'discomfort': 31, 'innovativeness': 19}
# recall: {'discomfort': 0.4, 'innovativeness': 0.0, 'insecurity': 0.3333333333333333, 'optimism': 0.0}
# f-measure: {'discomfort': 0.3076923076923077, 'innovativeness': 0, 'insecurity': 0.3076923076923077, 'optimism': None}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="SVM",
                                language="spanish", stem=False, train_method="all_class_train")

# accuracy: 0.2222222222222222
# precision: {'discomfort': 0.125, 'innovativeness': 0.0, 'insecurity': 0.4, 'optimism': 0.25}
# num of categories: {'insecurity': 35, 'optimism': 19, 'discomfort': 31, 'innovativeness': 17}
# recall: {'discomfort': 0.2, 'innovativeness': 0.0, 'insecurity': 0.3333333333333333, 'optimism': 0.25}
# f-measure: {'discomfort': 0.15384615384615385, 'innovativeness': 0, 'insecurity': 0.36363636363636365, 'optimism': 0.25}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="NB",
                                language="spanish", stem=False, train_method="cross_validation")

# accuracy: 0.5
# precision: {'discomfort': 1.0, 'innovativeness': 0.0, 'insecurity': 1.0, 'optimism': 0.4}
# num of categories: {'insecurity': 46, 'optimism': 16, 'discomfort': 24, 'innovativeness': 16}
# recall: {'discomfort': 0.5, 'innovativeness': None, 'insecurity': 0.3333333333333333, 'optimism': 0.6666666666666666}
# f-measure: {'discomfort': 0.6666666666666666, 'innovativeness': None, 'insecurity': 0.5, 'optimism': 0.5}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="DT",
                                language="spanish", stem=False, train_method="cross_validation")

# accuracy: 0.5
# precision: {'discomfort': 1.0, 'innovativeness': 0.5, 'insecurity': 0.5, 'optimism': 0.0}
# num of categories: {'insecurity': 45, 'optimism': 17, 'discomfort': 23, 'innovativeness': 17}
# recall: {'discomfort': 0.3333333333333333, 'innovativeness': 1.0, 'insecurity': 0.6666666666666666, 'optimism': 0.0}
# f-measure: {'discomfort': 0.5, 'innovativeness': 0.6666666666666666, 'insecurity': 0.5714285714285714, 'optimism': 0}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="RF",
                                language="spanish", stem=False, train_method="cross_validation")

# accuracy: 0.375
# precision: {'discomfort': 0.16666666666666666, 'innovativeness': None, 'insecurity': 1.0, 'optimism': None}
# num of categories: {'insecurity': 34, 'optimism': 16, 'discomfort': 35, 'innovativeness': 17}
# recall: {'discomfort': 1.0, 'innovativeness': None, 'insecurity': 0.5, 'optimism': 0.0}
# f-measure: {'discomfort': 0.2857142857142857, 'innovativeness': None, 'insecurity': 0.6666666666666666, 'optimism': None}

classifier = DocumentClassifier(train_p=0.8, eq_label_num=True, complete_p=True,
                                n_folds=10, vocab_size=250, t_classifier="SVM",
                                language="spanish", stem=False, train_method="cross_validation")

# accuracy: 0.75
# precision: {'discomfort': 0.3333333333333333, 'innovativeness': 1.0, 'insecurity': 1.0, 'optimism': None}
# num of categories: {'insecurity': 38, 'optimism': 21, 'discomfort': 27, 'innovativeness': 16}
# recall: {'discomfort': 1.0, 'innovativeness': 0.6666666666666666, 'insecurity': 1.0, 'optimism': 0.0}
# f-measure: {'discomfort': 0.5, 'innovativeness': 0.8, 'insecurity': 1.0, 'optimism': None}


predictions = classifier.classify_docs(input_set)
classifiedd = classifier._classified_docs
accur = classifier._accuracy
preci = classifier._precision
reca = classifier._recall
f_meas = classifier._f_measure
categ_num = classifier._final_cat_count
print(classifiedd) 
print(accur)
print(preci)
print(categ_num)
print(reca)
print(f_meas)



import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline



data = input_set

# Load the labeled data into a pandas dataframe
data = pd.DataFrame(data, columns=['responding', 'categories'])
print(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Convert the text data into a bag-of-words representation
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('spanish'))
train_vectors = vectorizer.fit_transform(train_data['responding'])
test_vectors = vectorizer.transform(test_data['responding'])

# Train and evaluate multiple classifiers
classifiers = [
    MultinomialNB(),
    LinearSVC(),
    RandomForestClassifier()
]

for clf in classifiers:
    clf.fit(train_vectors, train_data['categories'])
    scores = cross_val_score(clf, train_vectors, train_data['categories'], cv=5)
    print(f'{clf.__class__.__name__} cross-validation scores: {scores}')
    predicted_labels = clf.predict(test_vectors)
    print(f'{clf.__class__.__name__} classification report:\n{classification_report(test_data["categories"], predicted_labels)}\n')

# Choose the best classifier based on the metrics
best_clf = max(classifiers, key=lambda clf: np.mean(cross_val_score(clf, train_vectors, train_data['categories'], cv=5)))
print(f'Best classifier: {best_clf.__class__.__name__}')



with open('algoritmo_trained_set.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    # create formatted list of tuples
    formatted_list = [(row[0], row[1]) for row in reader]
    for row in formatted_list:
        print(row)


data = formatted_list
print(data)

# Load the labeled data into a pandas dataframe
data = pd.DataFrame(formatted_list, columns=['responding', 'categories'])
print(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.6)
train_data = data[:train_size]
test_data = data[train_size:]

# Convert the text data into a vector representation
ngram_range= (1, 1)
max_df=0.8
min_df=1

vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('spanish'), ngram_range=ngram_range, max_df=max_df, min_df=min_df)
train_vectors = vectorizer.fit_transform(train_data['responding'])
test_vectors = vectorizer.transform(test_data['responding'])

# Train and evaluate multiple classifiers
classifiers = [MultinomialNB(alpha=0.5), LinearSVC(C=1.0), 
               RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42), 
               KNeighborsClassifier(n_neighbors=7, weights='uniform'),
               DecisionTreeClassifier(max_depth=15, random_state=42),
               GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
]

for clf in classifiers:
    param_grid = {}
    if clf.__class__.__name__ == 'MultinomialNB':
        param_grid = {'alpha': [0.1, 0.5, 1.0]}
    elif clf.__class__.__name__ == 'LinearSVC':
        param_grid = {'C': [0.1, 1.0, 10.0]}
    elif clf.__class__.__name__ == 'RandomForestClassifier':
        param_grid = {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, None]}
    elif clf.__class__.__name__ == 'KNeighborsClassifier':
        param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']} 
    elif clf.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = {'max_depth': [3, 5, 15, 50], 'random_state': [42]} 
    elif clf.__class__.__name__ == 'GradientBoostingClassifier':
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0], 'max_depth': [3, 5, 10], 'random_state': [42]}
    
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(train_vectors, train_data['categories'])
    clf = grid_search.best_estimator_
    scores = cross_val_score(clf, train_vectors, train_data['categories'], cv=5)
    print(f'{clf.__class__.__name__} cross-validation scores: {scores}')
    predicted_labels = clf.predict(test_vectors)
    print(f'{clf.__class__.__name__} classification report:\n{classification_report(test_data["categories"], predicted_labels)}\n')


# Use an ensemble method to combine the predictions of the classifiers
voting_clf = VotingClassifier(estimators=[('nb', classifiers[0]), ('svc', classifiers[1]), ('rf', classifiers[2]), ('knn', classifiers[3]), ('dt', classifiers[4]), ('gb', classifiers[5])])
voting_clf.fit(train_vectors, train_data['categories'])
predicted_labels = voting_clf.predict(test_vectors)
print(f'Ensemble classifier classification report:\n{classification_report(test_data["categories"], predicted_labels)}\n')


data = formatted_list
print(data)

# Load the labeled data into a pandas dataframe
data = pd.DataFrame(formatted_list, columns=['responding', 'categories'])
print(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.6)
train_data = data[:train_size]
test_data = data[train_size:]

# Convert the text data into a vector representation
ngram_range= (1, 1)
max_df=0.8
min_df=1

vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('spanish'), ngram_range=ngram_range, max_df=max_df, min_df=min_df)
train_vectors = vectorizer.fit_transform(train_data['responding'])
test_vectors = vectorizer.transform(test_data['responding'])

# Define the classifiers to be used
classifiers = [
    MultinomialNB(),
    LinearSVC(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GradientBoostingClassifier()
]

# Define hyperparameters for each classifier
hyperparameters = {
    'MultinomialNB': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},#
    'LinearSVC': {'C': [0.1, 1.0, 10.0, 100.0]},
    'RandomForestClassifier': {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20, None]},#
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}, #
    'DecisionTreeClassifier': {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]}, #
    'GradientBoostingClassifier': {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 10]} #
}

# Initialize variables for storing best hyperparameters and scores
best_params = {}
best_scores = {}

# Iterate through each classifier
for clf in classifiers:
    clf_name = clf.__class__.__name__
    param_grid = hyperparameters.get(clf_name)
    if param_grid:
        # Perform grid search to find the best combination of hyperparameters
        grid_search = GridSearchCV(clf, param_grid, cv=5)
        grid_search.fit(train_vectors, train_data['categories'])
        # Store the best hyperparameters and scores
        best_params[clf_name] = grid_search.best_params_
        best_scores[clf_name] = grid_search.best_score_
        # Train and evaluate the classifier using the best hyperparameters
        clf.set_params(**grid_search.best_params_)
    else:
        # Train and evaluate the classifier using the default hyperparameters
        best_scores[clf_name] = cross_val_score(clf, train_vectors, train_data['categories'], cv=5).mean()
    scores = cross_val_score(clf, train_vectors, train_data['categories'], cv=5)
    print(f'{clf_name} cross-validation scores: {scores}')
    clf.fit(train_vectors, train_data['categories'])
    predicted_labels = clf.predict(test_vectors)
    print(f'{clf_name} classification report:\n{classification_report(test_data["categories"], predicted_labels)}\n')

# Print the best hyperparameters and scores for each classifier
print('Best hyperparameters and scores:')
for clf_name in best_params:
    print(f'{clf_name}: {best_params[clf_name]} - {best_scores[clf_name]}')

# Use a voting ensemble to combine the predictions of the classifiers
voting_clf = VotingClassifier(estimators=[('nb', classifiers[0]), ('svc', classifiers[1]), ('rf', classifiers[2]), ('knn', classifiers[3]), ('dt', classifiers[4]), ('gb', classifiers[5])])
voting_clf.fit(train_vectors, train_data['categories'])
predicted_labels = voting_clf.predict(test_vectors)
print(f'Ensemble classifier classification report:\n{classification_report(test_data["categories"], predicted_labels)}\n')






# Import necessary libraries
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import ParameterGrid

data = []
for row in formatted_list: 
    tweet = row[0]
    tokenize_and_stop_wordsi = tokenize_and_remove_stop_words(tweet, language='spanish', join_words=True)
    tokenize_and_stemi = tokenize_and_stem(tokenize_and_stop_wordsi, language='spanish', join_words=True)
    format_data = (tokenize_and_stemi, row[1])
    data.append(format_data)
print(data)

# Load the labeled data into a pandas dataframe
data = pd.DataFrame(data, columns=['responding', 'categories'])
print(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.6)
train_data = data[:train_size]
test_data = data[train_size:]

# Convert the text data into a vector representation
ngram_range= (1, 1)
max_df=0.8
min_df=1

vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('spanish'), ngram_range=ngram_range, max_df=max_df, min_df=min_df)
train_vectors = vectorizer.fit_transform(train_data['responding'])
test_vectors = vectorizer.transform(test_data['responding'])

# Define the classifiers to be used
classifiers = [
    MultinomialNB(),
    LinearSVC(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GradientBoostingClassifier()
]

# Define hyperparameters for each classifier
hyperparameters = {
    'MultinomialNB': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},#
    'LinearSVC': {'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'RandomForestClassifier': {'n_estimators': [200, 500, 1000, 2000], 'max_depth': [5, 10, 20, 30, None]},#
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 9, 11, 13], 'weights': ['uniform', 'distance']}, #
    'DecisionTreeClassifier': {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10, 20, 50], 'splitter': ['best', 'random']}, #
    'GradientBoostingClassifier': {'n_estimators': [200, 500, 1000, 2000], 'learning_rate': [0.001, 0.01, 0.1, 1.0], 'max_depth': [3, 5, 10]} #
}


hyperparameters = {
    'MultinomialNB': {'alpha': [10 ** i for i in range(-4, 2)]},
    'LinearSVC': {'C': [10 ** i for i in range(-2, 4)]},
    'RandomForestClassifier': {'n_estimators': [200, 500, 1000, 2000], 'max_depth': range(1, 11)},
    'KNeighborsClassifier': {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']},
    'DecisionTreeClassifier': {'max_depth': range(1, 11), 'min_samples_split': [2, 5, 10, 20, 50], 'splitter': ['best', 'random']},
    'GradientBoostingClassifier': {'n_estimators': [200, 500, 1000, 2000], 'learning_rate': [0.001, 0.01, 0.1, 1.0], 'max_depth': range(1, 11)}
}

hyperparameters = {
    'MultinomialNB': {'alpha': [10 ** i for i in range(-4, 2)]},
    'LinearSVC': {'C': [10 ** i for i in range(-2, 4)]},
    'RandomForestClassifier': {'n_estimators': list(range(200, 2001, 300)), 'max_depth': list(range(1, 11))},
    'KNeighborsClassifier': {'n_neighbors': list(range(1, 21)), 'weights': ['uniform', 'distance']},
    'DecisionTreeClassifier': {'max_depth': list(range(1, 11)), 'min_samples_split': list(range(2, 51, 8)), 'splitter': ['best', 'random']},
    'GradientBoostingClassifier': {'n_estimators': list(range(200, 2001, 300)), 'learning_rate': [0.001, 0.01, 0.1, 1.0], 'max_depth': list(range(1, 11))}
}


# Define a function to find the best hyperparameters for a given classifier
def find_best_params(clf_name, clf, hyperparameters, train_vectors, train_labels):
    param_grid = hyperparameters.get(clf_name)
    if param_grid:
        # Perform grid search to find the best combination of hyperparameters
        grid_search = GridSearchCV(clf, param_grid, cv=5)
        grid_search.fit(train_vectors, train_labels)
        # Train and evaluate the classifier using the best hyperparameters
        clf.set_params(**grid_search.best_params_)
        scores = cross_val_score(clf, train_vectors, train_labels, cv=5)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
    else:
        # Train and evaluate the classifier using the default hyperparameters
        scores = cross_val_score(clf, train_vectors, train_labels, cv=5)
        best_params = {}
        best_score = scores.mean()
    return best_params, best_score

# Initialize variables for storing best hyperparameters and scores
best_params = {}
best_scores = {}

def get_classifier_name(clf):
    if isinstance(clf, MultinomialNB):
        return 'MultinomialNB'
    elif isinstance(clf, LinearSVC):
        return 'LinearSVC'
    elif isinstance(clf, RandomForestClassifier):
        return 'RandomForestClassifier'
    elif isinstance(clf, KNeighborsClassifier):
        return 'KNeighborsClassifier'
    elif isinstance(clf, DecisionTreeClassifier):
        return 'DecisionTreeClassifier'
    elif isinstance(clf, GradientBoostingClassifier):
        return 'GradientBoostingClassifier'
    else:
        return 'UnknownClassifier'

# Iterate through each classifier
for i, clf in enumerate(classifiers):
    clf_name = get_classifier_name(clf)
    best_score = 0

    # Iterate through hyperparameters for the current classifier
    for params in ParameterGrid(hyperparameters[clf_name]):
        clf.set_params(**params)
        scores = cross_val_score(clf, train_vectors, train_data['categories'], cv=5)
        mean_score = np.mean(scores)
        
        # Update best hyperparameters and score if current score is higher than previous best score
        if mean_score > best_score:
            best_score = mean_score
            best_params[clf_name] = params

        # Break out of loop if best score is already at least 0.60
        if best_score >= 0.60:
            break

    best_scores[clf_name] = best_score

for clf_name in best_params.keys():
    print(f"{clf_name}:")
    print(f"Best hyperparameters: {best_params[clf_name]}")
    print(f"Best score: {best_scores[clf_name]}")



print('hello')