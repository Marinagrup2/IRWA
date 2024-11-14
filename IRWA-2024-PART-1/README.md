# IRWA

PART 1

1. Pre-processing

We chose to do our project with google colab, so the execution of the project could be easy and understandable
To pre-process the tweet documents, we first have to connect to the drive, import the diverse libriaries used 
and the tweets/ doc_ids to use this data

As the tweet data contains a wide variety of information, we decided to keep a few keys of just the information
the user will see when entering a query and store this in a new array "tweets_values_to_keep"
Then we map the tweet ids to the ones we imported from doc_ids and elimitate the ones that are not in doc_id

In the function preprocess_tweet, we eliminate from the key "content" the various elements we don´t want in the text
such as mentions, hashtags, urls to the pictures or videos the tweet might have, any extra spaces there might me 
or the emojis some tweets have (calling the function remove_emojis). We also create a new key for the hastags we 
removed from the content to store them separately without the simbol "#" 

Finally, we call the function build_terms where we transform the text to lowercase, remove punctuation,
tokenize, eliminate stopwords and perform stemming after combining the content and the hashtags (without the simbol)
so we have both keys in the same line to pass to the build_terms and we can take into account the hashtags
while also keeping them separately from the tweet content

2. Exploratory Data Analysis

Word count and top words:
The objective is to print the top 10 most frequent words in a collection of tweets. To achieve this, we do the following process:
content_terms = content_words():
There the function content words is called. This function iterates through all the tweets in tweets_separated_info, a set that we have filtered before by removing mentions, hashtags, URLs, extra spaces, and emojis. If the tweet has content, we build the terms by tokenizing them, eliminating the stopwords, and performing stemming. We do an update on the Counter word_counts, that will add every token to the set if it’s not in there and start a new count or add 1 to the count if the token is already there. The function will return that set.
top_10_words = content_terms.most_common(10)
This returns the 10 elements from the set with the higher count.
print("Top 10 most frequent words:")
for word, count in top_10_words:
  print(f"{word}: {count}")
Finally, we print it.

Word clouds for the most frequent words:
The objective is to represent in word clouds the most frequent words in the tweets we have done in the word count and store them in the variable content_terms. For that we have created the function plot_top_x_words_wordcloud(words, x) where words is a Counter and x is an integer.
The function grabs the most_common x words and then plot the WordCloud. Since we sent the function the content_terms and x = 50, the word cloud will show the most common 50 words of the content-filtered tweets.

Average sentence length:
The objective is to compute the average sentence length of all tweets. For that, we create the function calculate_avg_sentence_length_for_collection(tweets). We will compute the average length of the tweets individually with the function average_sentence_length_for_tweet(content) and add it to the variable total_length. This variable will end up having the total length of the sum of all the tweets. The average lenth will be dividing this with the number of tweets we have stored in the variable total_tweets. 
When we call the function we send the variable tweets_separated_info, which are the filtered tweets.

Top hashtags:
The objective is to find the top hashtags. To do so, we have created the function find_top_hashtags(tweet_dict, top_n) where tweet_dic is the dictionary of tweets and top_n is the number that we want to know the top of. 
For each tweet in the dictionary, we select only the hashtags. From that list, we create a Counter that will add every hashtag to the set if it’s not in there and start a new count or add 1 to the count if the token is already there. From there. we select the most common top_n words.
When we call this function we will send tweets_values_to_keep, which contains the hashtags. Then we create the cloudplot. 

Most retweeted/ liked:
We create a function for each request. 
To get the most retweeted tweets we create the function get_most_retweeted(tweets, n) where tweets are all the tweets and n is to get the most top n retweeted tweets. 
To get the most liked tweets we create the function get_most_liked(tweets, n) where tweets are all the tweets and n is to get the most top n liked tweets. 

Both functions sort the tweets based on the value of the ‘retweetCount’ or ‘likeCount’ key in each dictionary, in descending order. Then it selects the first n elements of the list. Then we print them.

