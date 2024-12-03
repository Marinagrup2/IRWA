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
import sqlite3
import gensim
from gensim.models import Word2Vec

  
def search_in_corpus(corpus, query, index, tf, idf, search_method):
    '''''
    Returns the ranked documents for query in corpus    
    '''''

    # 1. create create_tfidf_index
    # Done in web_app because it took too much time here
    print("Index created \n")

    # 2. apply ranking
    #ranked_tweets = search_tf_idf(query, index, tf, idf)
    ranked_tweets = search_score(query, index, tf, idf, corpus, search_method)

    print("Ranking done \n")

    return ranked_tweets


def create_index_tf_idf(corpus):
    total_tweets = len(corpus) # Total number of tweets
    print("Total number of tweets: {} \n".format(total_tweets))
    
    index = defaultdict(list)
    
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)   
    
    for page_id, tweet in corpus.items():
      print(page_id)
      print("\n")
      prep_tweet = preprocess_tweet(tweet.description)

      print(prep_tweet)

      current_page_index = defaultdict(lambda: [page_id, array('I')])

      # Populate current_page_index with term positions
      for position, term in enumerate(prep_tweet):
        current_page_index[term][1].append(position)
        
      # Calculate normalization factor (norm)
      norm = math.sqrt(sum(len(posting[1]) ** 2 for posting in current_page_index.values()))

      # Calculate term frequencies (tf) and document frequencies (df)
      for term, posting in current_page_index.items():
        index[term].append(posting)
        tf[term].append(len(posting[1]) / norm)
        df[term] += 1        

      # Compute IDF values using vectorized operations
      for term, doc_freq in df.items():
        idf[term] = np.log(total_tweets / doc_freq)

    return index, tf, idf  

def search_score(query, index, tf, idf, corpus, search_method):
    '''''
    Returns the ranked documents for query in corpus with a determined score   
    '''''
    query = preprocess_tweet(query)
    docs = set()
    for term in query:
        try:
            term_docs = [posting[0] for posting in index[term]]
            docs = docs.union(set(term_docs))
        except:
            pass
    docs = list(docs)

    # Choose the method
    
    if (search_method == "tf_idf_cosine_similarity"):
        # Method 1: tf_idf + cosine similarity
        ranked_docs = rank_documents_tf_idf_cosine_sim(query, docs, index, idf, tf)
    elif (search_method == "our_score_cosine_similarity"):
        # Method 2: Our score (likes and rt) + cosine similarity
        ranked_docs = rank_documents_your_score(query, docs, index, idf, tf, corpus) 
    elif (search_method == "BM25"):
        # Method 3: BM25
        ranked_docs = rank_documents_bm25(query, docs, index, idf, tf, corpus)
    elif (search_method == "w2v_cosine_similarity"):
        # Method 4: Word2vec
        ranked_docs = rank_documents_word2vec(query, corpus)
    else:
        # If not selected any, default is the method 1
        ranked_docs = ranked_docs = rank_documents_tf_idf_cosine_sim(query, docs, index, idf, tf)

    return ranked_docs

def search_tf_idf(query, index, tf, idf):
    """
    Output is the list of documents that contain any of the query terms.
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
    ranked_docs = rank_documents_tf_idf_cosine_sim(query, docs, index, idf, tf)
    return ranked_docs

def rank_documents_tf_idf_cosine_sim(terms, docs, index, idf, tf):
    """
    Return the ranked docs based of tf-idf + cosine sim score
    """
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
        for doc_index, (doc, _) in enumerate(index[term]):

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate the score of each doc with cosine similarity
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]

    # In case no docs are found
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)

    return result_docs

def rank_documents_your_score(terms, docs, index, idf, tf, corpus):
    """
    Return the ranked docs based of tf-idf + cosine sim score
    """
    # Prepare the vectors for the cosine similarity
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    # Compute the query norm based on the term frequency of the terms of the query
    query_terms_count = collections.Counter(terms) # Counter for each term on the query (tf)
    query_norm = la.norm(list(query_terms_count.values())) # Euclidean norm (magnitude) of the query vector using the term frequencies.

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        # Calculate the weight of the term in the query vector using TF-IDF
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        for doc_index, (doc, _) in enumerate(index[term]):
            # If the current document is in the set of relevant documents for the query
            if doc in docs:
                # Calculate the weight of the term in the document vector using TF-IDF
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    doc_scores = []
    for doc_id, doc_vec in doc_vectors.items():
        # Compute the cosine similarity
        cosine_sim = cosine_similarity(doc_vec, query_vector)

        # Compute the "our score" 
        like_score = np.log(corpus[doc_id].likes + 1) # Like score
        retweet_score = np.log(corpus[doc_id].retweets + 1) # Retweet score

        # Compute the final score by combining the 3 scores
        your_score = 0.7 * cosine_sim + 0.2 * like_score + 0.2 * retweet_score
        doc_scores.append([your_score, doc_id])

    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    return result_docs

def rank_documents_bm25(terms, docs, index, idf, tf, corpus, k1=1.5, b=0.75):

    doc_scores = defaultdict(float)
    doc_lengths, avg_doc_length= calculate_doc_lengths_and_avg(corpus)
    
    # Iterate over each term in the query
    for term in terms:
        if term in index:

          for doc_index, (doc, _) in enumerate(index[term]):
              if doc in docs:

                # Get the term frequency of `term` in the document
                term_frequency = tf[term][doc_index]

                # Document length normalization
                doc_length = doc_lengths[doc]
                norm_term_frequency = ((term_frequency * (k1 + 1)) / (term_frequency + k1 * (1 - b + b * doc_length / avg_doc_length)))

                # Calculate BM25 score component for this term in this document
                doc_scores[doc] += idf[term] * norm_term_frequency

    # Sort documents by score in descending order
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    result_docs = [doc_id for doc_id, _ in ranked_docs]

    return result_docs

def rank_documents_word2vec(query, corpus):
    """
    Ranks documents based on cosine similarity between tweet2vec and query2vec.
    """
    model = model_word2vec(corpus)

    query_vector = tweet2vec(query, model) # Generates a vector representation of the query
    doc_scores = []

    for doc_id, tweet in corpus.items():  # Iterate over each document
      doc_vector = tweet2vec(tweet.description, model) # Generates a vector representation of the tweet

      # Computing doc scores based on cosine similarity
      score = cosine_similarity(query_vector, doc_vector)
      doc_scores.append((doc_id, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [doc_id for doc_id, _ in doc_scores]
    
    return ranked_docs

# Other functions to help

def preprocess_tweet(tweet):    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    tweet = tweet.lower() # Transform in lowercase
    tweet = re.sub(r'[{}]'.format(string.punctuation), '', tweet) # Remove punctuation
    words = tweet.split() # Tokenize the text to get a list of terms
    processed_words = [word for word in words if word not in stop_words] # Eliminate the stopwords
    processed_words = [stemmer.stem(word) for word in processed_words] # Perform stemming     
    
    return processed_words

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = la.norm(v1)
    norm_v2 = la.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Handle cases where either vector has zero magnitude
    else:
        return dot_product / (norm_v1 * norm_v2)
    
def calculate_doc_lengths_and_avg(corpus):

    doc_lengths = {}
    total_length = 0

    # Calculate each document's length and the total length
    for doc_id, tweet in corpus.items():
        doc_length = len(tweet.description.split())  # Number of words in content
        doc_lengths[doc_id] = doc_length
        total_length += doc_length

    # Calculate the average document length
    avg_doc_length = total_length / len(corpus) if corpus else 0

    return doc_lengths, avg_doc_length

def model_word2vec(corpus):
  """Generates the model for word2vec."""
  tweets = [tweet.description for _, tweet in corpus.items()]
  words = [tweet.split() for tweet in tweets]
  model = Word2Vec(sentences=words, workers=4, window=3, min_count=200)
  return model

def tweet2vec(tweet, model):
    """
    Generates a tweet representation (tweet2vec) by averaging word vectors.
    """
    vectors = []
    for word in tweet:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            # Handle words not found in the vocabulary (e.g., using a zero vector)
            vectors.append(np.zeros(model.vector_size))

    #Return the avg of the vectors or a zero vector for empty tweets
    if vectors:
        return np.mean(vectors, axis=0)
    else:
      return np.zeros(model.vector_size)
