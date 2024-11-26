import pandas as pd

from myapp.core.utils import load_json_file
from myapp.search.objects import Document

from pandas import json_normalize
import json
import re

_corpus = {}


def load_corpus(path) -> [Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = _load_corpus_as_dataframe(path)
    df.apply(_row_to_doc_dict, axis=1)
    return _corpus


def _load_corpus_as_dataframe(path):
    """
    Load documents corpus from file in 'path'
    :return:
    """
    data_list = []

    #read all lines
    with open(path) as f:
        for line in f:
            # Parse the string into a JSON object
            json_data = json.loads(line)
            # Append the JSON object to the list
            data_list.append(json_data)

    # Convert the list of JSON objects to a DataFrame
    df = pd.DataFrame(data_list)
    
    for index, row in df.iterrows():
        user_name = row['user']['username'] # Access the 'username' key
        profile_pic = row['user']['profileImageUrl']
        tweet_id = row['id']
        tweet_content = row['content']
        hashtags = [tag[1:] for tag in re.findall(r'#\w+', tweet_content) for _ in range(3)]

        # Create the 'Url' and 'Hashtags' column for each row
        df.at[index, "Url"] = f"https://twitter.com/{user_name}/status/{tweet_id}"
        df.at[index, "Hashtags"] = len(hashtags)
        df.at[index, "Profile_pic"] = profile_pic
        df.at[index, "Username"] = user_name

    df = df.rename(columns={
        "id": "Id",
        "content": "Tweet",
        "username": "Username",
        "date": "Date",
        "likeCount": "Likes",
        "retweetCount": "Retweets",
        "lang": "Language"
    })
    return df

def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Document(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'], row['Likes'],
                                  row['Retweets'],
                                  row['Url'], row['Hashtags'])
