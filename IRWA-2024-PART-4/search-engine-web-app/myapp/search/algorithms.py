import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string
import collections
from collections import defaultdict
from array import array
import math
import numpy as np
from numpy import linalg as la

  
def search_in_corpus(corpus, query):
    # 1. create create_tfidf_index
    index, tf, idf = create_index_tf_idf(corpus)

    # 2. apply ranking
    ranked_tweets = search_tf_idf(query, index, tf, idf)

    return ranked_tweets

def create_index_tf_idf(corpus):
    total_tweets = len(corpus) # Total number of tweets
    index = defaultdict(list)
    
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)   
    
    for page_id, row in corpus.items():
      prep_tweet = preprocess_tweet(row.description)
      current_page_index = defaultdict(lambda: [page_id, array('I')])
      print(current_page_index) 
      print("\n") 

      # Populate current_page_index with term positions
      for position, term in enumerate(prep_tweet):
        current_page_index[term][1].append(position)
        
      # Calculate normalization factor (norm)
      norm = math.sqrt(sum(len(posting[1]) ** 2 for posting in current_page_index.values()))

      # Calculate term frequencies (tf) and document frequencies (df)
      for term, posting in current_page_index.items():
        tf[term].append(len(posting[1]) / norm)
        df[term] += 1

      # Merge current page index with the main index
      for term, posting in current_page_index.items():
        index[term].append(posting)

      # Compute IDF values using vectorized operations
      for term, doc_freq in df.items():
        idf[term] = np.log(total_tweets / doc_freq)

    return index, tf, idf  

def search_tf_idf(query, index, tf, idf):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = preprocess_tweet(query)
    docs = set()
    
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs = docs.union(set(term_docs))
        except:
            #term is not in index
            pass
    
    docs = list(docs)
    ranked_docs = rank_documents(query, docs, index, idf, tf)
    return ranked_docs

def rank_documents(terms, docs, index, idf, tf):
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # Documents vectors empty
    query_vector = [0] * len(terms) # Query vectors empty

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm * idf[term] #frequency of term normalized *idf of the term

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate the score of each doc with cosine similarity
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]

    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)

    return result_docs

def clean_tweet(tweet_content):
    cleaned_content = re.sub(r'@\w+', '', tweet_content)  # Remove mentions
    cleaned_content = re.sub(r'#\w+', '', cleaned_content)  # Remove hashtags
    cleaned_content = re.sub(r'http\S+', '', cleaned_content)  # Remove URLs
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()  # Remove extra spaces
    cleaned_content = remove_emojis(cleaned_content)

    return cleaned_content

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

def preprocess_tweet(tweet):
    
    #tweet = clean_tweet(tweet)
    #print(tweet)
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    tweet = tweet.lower() # Transform in lowercase

    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Remove extra spaces
    tweet = re.sub(r'[{}]'.format(string.punctuation), '', tweet) # Remove punctuation

    words = tweet.split() # Tokenize the text to get a list of terms
    processed_words = [word for word in words if word not in stop_words] # Eliminate the stopwords
    processed_words = [stemmer.stem(word) for word in processed_words] # Perform stemming     
    
    return processed_words