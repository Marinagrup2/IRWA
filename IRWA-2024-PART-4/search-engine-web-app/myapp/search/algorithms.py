import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string
import collections
from collections import defaultdict

def remove_emojis(text):

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r' ', text)  # Replace emojis by an empty string

def clean_corpus(corpus):

  for i, tweet in enumerate(corpus):
    tweet_content = corpus[i]['description']

    cleaned_content = re.sub(r'@\w+', '', tweet_content)  # Remove mentions
    cleaned_content = re.sub(r'#\w+', '', cleaned_content)  # Remove hashtags
    cleaned_content = re.sub(r'http\S+', '', cleaned_content)  # Remove URLs
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()  # Remove extra spaces
    cleaned_content = remove_emojis(cleaned_content)

    corpus[i]['description'] = cleaned_content

  return corpus

def preprocess_tweet(tweet):

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    tweet = tweet.lower() # Transform in lowercase
    tweet = re.sub(r'[{}]'.format(string.punctuation), '', tweet) # Remove punctuation
    words = tweet.split() # Tokenize the text to get a list of terms
    processed_words = [word for word in words if word not in stop_words] # Eliminate the stopwords
    processed_words = [stemmer.stem(word) for word in processed_words] # Perform stemming     
    
    return ' '.join(processed_words)


def create_index_tf_idf(corpus):
    total_tweets = len(corpus) # Total number of tweets
    index = defaultdict(list)
    
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)

    # We eliminate mentions, hashtags, URLs and extra spaces from the tweets description
    clean_corpus = clean_corpus(corpus)   
    
    #Falta por mirar
    for page_id, row in corpus.items():
       prep_tweet = preprocess_tweet(row.description)
       
       
        



def search_in_corpus(query):
    # 1. create create_tfidf_index

    # 2. apply ranking
    return ""
