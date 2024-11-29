import pandas as pd

from myapp.core.utils import load_json_file
from myapp.search.objects import Document

#from pandas import json_normalize
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
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    
    data = pd.DataFrame(data)
    
    corpus = data.rename(columns={
        "id": "Id",
        "content": "Tweet",
        "date": "Date",
        "likeCount": "Likes",
        "retweetCount": "Retweets",
        "hashtags": "Hashtags",
        "url": "Url"
    })
    
    return corpus

def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Document(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'], row['Likes'],
                                  row['Retweets'],
                                  row['Url'], row['Hashtags'])
