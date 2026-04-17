########## 1. Import required libraries ##########
import pandas as pd
import numpy as np
import re
import math

#Using the model.
import joblib

# Text cleaning & stopwords
import nltk
#nltk library to download the stopwords array.
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    #compile the regular expression to search for any parts that have the structure < ... >
    html = re.compile(r'<.*?>')
    # Replace all occurrences of " < ... > " with "" to remove the html brackets
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
#imported stopwords list for english
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    #some list copmprehension for removing stop words from the given string
    #inside the list, the if word not in final_stop_words_list, is used to joing the resulting string the next word if it is NOT a stop word.
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Load the models & preprocess input. ##########
import os
import subprocess

tfidf = joblib.load('vectorizer.joblib')
classifier = joblib.load('model.joblib')


#get the report from the user
title = input("Title of the report:")
body = input("Body of the report:")

com_in = title + "\n" + body

#Preprocess text
no_html = remove_html(com_in)
no_emoji = remove_emoji(no_html)
no_stopw = remove_stopwords(no_emoji)
text = clean_str(no_stopw)

tokens = tfidf.transform([text])
result = classifier.predict(tokens)[0]
result_pro = classifier.predict_proba(tokens)[0][result]

if result:
    print(f"The reported bug was a performance related bug. (confidence {result_pro * 100:.1f}%)")
else:
    print("Not a performance related bug.")