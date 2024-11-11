# IRWA

PROJECT:
We start by processing the tweet data with the functions we did in the Part 1 of the the Project. We also added to our preprocess_tweet function the repetition of hashtags we mentioned on part1 so those have more weight and the importance of the hashtags is present. 

We create the inverted index using the function create_index_tfidf. This function stores each term in the dataset alongside a list of documents (doc_ids) where the term appears, includes the position of each term within the document, which allows for accurate frequency 
and position tracking and calculates the TF (Term Frequency) and IDF (Inverse Document Frequency), following the approach discussed in the course.
Previously, there are 2 other functions (create_index), one that does the index of each word without the positions in the document and the other with the positions in the document. These are not used for our searching but we left them to find the docs that contain the words faster without having the positions or waiting for the idf to be calculated.
The function search_tf_idf identifies documents that contain all query terms, ranking these documents by relevance to allow users to see the most pertinent tweets.

Our queries are designed based on the top 20 most frequent words in the dataset. After analyzing these frequent terms, we selected the following queries to reflect common themes in the data:

Farmers protest
Farmers rights
India is being silenced
Detained farmers
Support farmers revolution

The results for each query return a list of ranked tweets, ordered by relevance score. This demonstrates the TF-IDF index and ranking approach.

For the evaluation, we separated the functions of each evaluation technique. To test them, we firstly defined the ground truth of our queries, and for each, calculated from our functions the results.

Finally, to run the scatter plot, run the last cell where we calculate the tfidf matrix of our tweets (the content) and then apply T-SNE to the matrix to plot all the tweets with their TF-IDF weight

HOW TO RUN CODE:
To run the code, upload the notebook to Google Drive, open it with Google Colab and execute all the cells from the beginning to the end.