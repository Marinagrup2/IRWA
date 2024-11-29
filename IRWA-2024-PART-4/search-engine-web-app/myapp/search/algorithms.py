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

  
def search_in_corpus(corpus, query, index, tf, idf, analytics_db):
    # 1. create create_tfidf_index
    # Done in web_app because it took too much time here

    print("Index created \n")

    # 2. apply ranking
    #ranked_tweets = search_tf_idf(query, index, tf, idf)
    ranked_tweets = search_your_score(query, index, tf, idf, corpus, analytics_db)

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

def rank_documents_your_score(terms, docs, index, idf, tf, corpus, analytics_db):
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

        for doc_index, (doc, postings) in enumerate(index[term]):
            # If the current document is in the set of relevant documents for the query
            if doc in docs:
                # Calculate the weight of the term in the document vector using TF-IDF
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    doc_scores = []
    for doc, doc_vec in doc_vectors.items():
        # Compute the cosine similarity
        cosine_sim = cosine_similarity(doc_vec, query_vector)

        # This separates the "doc_123" to ['doc', '123'] (we grab only the number i.e. the id)
        doc_id = int(doc.split('_')[1])
        #print(title_index[doc].likes)
        like_score = np.log(corpus[doc].likes + 1) # Like score
        retweet_score = np.log(corpus[doc].retweets + 1) # Retweet score
        #like_score = np.log(title_index[doc_id].get('likeCount', 0) + 1) # Like score
        #retweet_score = np.log(title_index[doc_id].get('retweetCount', 0) + 1) # Retweet score

        doc_clicks = 0
        
        #Take into account clicks
        #query = "SELECT COUNT(*) FROM analytics WHERE doc_id IS " + doc
        '''
        with sqlite3.connect(analytics_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analytics IF EXISTS (SELECT * FROM analytics WHERE doc_id IS " + doc + ")")
            cursor.execute("SELECT COUNT(*) FROM analytics WHERE doc_id IS " + doc + " OR '1'='0'")
            if cursor.fetchone()[0] is False:
                doc_clicks = cursor.fetchone()[0]
        '''

        # Compute the final score by combining the 3 scores
        your_score = 0.7 * cosine_sim + 0.2 * like_score + 0.2 * retweet_score + 0.5*doc_clicks
        doc_scores.append([your_score, doc])

    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    return result_docs

def search_your_score(query, index, tf, idf, corpus, analytics_db):
    query = preprocess_tweet(query)
    docs = set()
    for term in query:
        try:
            term_docs = [posting[0] for posting in index[term]]
            docs = docs.union(set(term_docs))
        except:
            pass
    docs = list(docs)
    ranked_docs = rank_documents_your_score(query, docs, index, idf, tf, corpus, analytics_db)
    return ranked_docs