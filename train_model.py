########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import math

#Deploying the model.
import joblib

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.linear_model  import LogisticRegression

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

########## 3. Download & read data ##########
import os
import subprocess
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
#create list for projects and loop through it.
projects = ['pytorch','caffe','incubator-mxnet','keras','tensorflow']

trainings_x = []
tests_x = []
trainings_y = []
tests_y = []

for project in projects:
    path = f'datasets/{project}.csv'

    #using pandas, read every csv file into it's own dataframe
    pd_all = pd.read_csv(path)
    #frac = 1 (shuffle 100% of the rows) , random_state = 999 (specific shuffling method)
    pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle
    
    # Merge Title and Body into a single column; if Body is NaN, use Title only
    #creating new coloumn, .apply applies the function to given element (lambda is the name of the function) axis = 1 means to apply
    #the function row by row.
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )
    
    # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
    #.rename with coloumns parameter just reorganises the df such that only these coloumns remain.
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })

    data = pd_tplusb.fillna('')
    text_col = 'text'
    
    # Keep a copy for referencing original data if needed
    original_data = data.copy()
    
    # Text cleaning
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)
    
    # ========== Hyperparameter grid ==========
    # We use logspace for c: [0.01,0.1,1,10,100]
    params = {
        'C': np.logspace(-2, 2, 5)
    }
        
    # --- 4.1 Split into train/test ---
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    trainings_x.append(train_text)
    tests_x.append(test_text)
    trainings_y.append(y_train)
    tests_y.append(y_test)

# --- 4.2 TF-IDF vectorization ---
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1000  # Adjust as needed
)
X_train = tfidf.fit_transform(pd.concat(trainings_x))
X_test = tfidf.transform(pd.concat(tests_x))

Y_train = pd.concat(trainings_y)
Y_test = pd.concat(tests_y)

# --- 4.3 Logistic Regression ---
clf = LogisticRegression(random_state=0, class_weight='balanced')
grid = GridSearchCV(
    clf,
    params,
    cv=5,              # 5-fold CV (can be changed)
    scoring='roc_auc'  # Using roc_auc as the metric for selection
)
grid.fit(X_train, Y_train)

# --- 4.4 Make predictions & evaluate ---
y_pred = grid.predict(X_test)

# Accuracy
acc = accuracy_score(Y_test, y_pred)

# Precision (macro)
prec = precision_score(Y_test, y_pred, average='macro')

# Recall (macro)
rec = recall_score(Y_test, y_pred, average='macro')

# F1 Score (macro)
f1 = f1_score(Y_test, y_pred, average='macro')

# AUC
# If labels are 0/1 only, this works directly.
# If labels are something else, adjust pos_label accordingly.
fpr, tpr, _ = roc_curve(Y_test, y_pred, pos_label=1)
auc_val = auc(fpr, tpr)

print(f"=== Logistic Regression + TF-IDF Combined Results ===")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1 score:      {f1:.4f}")
print(f"AUC:           {auc_val:.4f}")

# Save final results to CSV (append mode)
out_csv_name = 'combined_results_LR.csv'
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'Accuracies': [acc],
        'Precisions': [prec],
        'Recalls': [rec],
        'F1s': [f1],
        'AUCs': [auc_val]
    }
)

df_log.to_csv(out_csv_name, mode='a', index=False)

print(f"\nResults have been saved to: {out_csv_name}")

joblib.dump(grid.best_estimator_, 'model.joblib')
joblib.dump(tfidf, 'vectorizer.joblib')
print(f"=== Model and the tfidf vectoriser deployed ===")