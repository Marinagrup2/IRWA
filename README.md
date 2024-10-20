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
