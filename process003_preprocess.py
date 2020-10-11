# import pandas as pd
# import os
# import random
#
# train_df=pd.read_csv("train.csv")
# test_df=pd.read_csv("test.csv")
#
# print(train_df.head())
# train_df = train_df[['id', 'sent0', 'sent1', 'label']]
#
# print(test_df.head())
# test_df = test_df[['id', 'sent0', 'sent1']]
#
# import re
# def replace_punctuation(example):
#     example = list(example)
#     pre = ''
#     cur = ''
#     for i in range(len(example)):
#         if i == 0:
#             pre = example[i]
#             continue
#         pre = example[i-1]
#         cur = example[i]
#         if re.match("[\u4e00-\u9fa5]", pre):
#             if re.match("[\u4e00-\u9fa5]", cur):
#                 continue
#             elif cur == ',':
#                 example[i] = '，'
#             elif cur == '.':
#                 example[i] = '。'
#             elif cur == '?':
#                 example[i] = '？'
#             elif cur == ':':
#                 example[i] = '：'
#             elif cur == ';':
#                 example[i] = '；'
#             elif cur == '!':
#                 example[i] = '！'
#             elif cur == '"':
#                 example[i] = '”'
#             elif cur == "'":
#                 example[i] = "’"
#     return ''.join(example)
#
# train_df['label']=train_df['label'].fillna(-1)
# train_df=train_df[train_df['label']!=-1]
# train_df['label']=train_df['label'].astype(int)
# test_df['label']=0
#
# test_df['sent1']=test_df['sent1'].fillna('.')
# train_df['sent1']=train_df['sent1'].fillna('.')
# test_df['sent0']=test_df['sent0'].fillna('.')
# train_df['sent0']=train_df['sent0'].fillna('.')
#
# rep_train_df = train_df[['sent0', 'sent1']].applymap(replace_punctuation)
# print(rep_train_df.head())
#
# rep_train_df = pd.concat([train_df[['id']],rep_train_df], axis=1)
# print(rep_train_df.head())
#
# rep_test_df = test_df[['sent0', 'sent1']].applymap(replace_punctuation)
# rep_test_df = pd.concat([test_df[['id']],rep_test_df], axis=1)
#
# train_title=[]
# test_title=[]
# test_content=[]
# train_content=[]
#
# # r1 = "[a-zA-Z'!\"#$%&'()*+,-./:;<=>?@★[\\]^_`{|}~]+"
# for train_str in rep_train_df['sent0']:
#     train_str = re.sub('url','',train_str)
#     train_title.append(train_str)
# for train_str in rep_train_df['sent1']:
#     train_str = re.sub(r1,'',train_str)
#     train_content.append(train_str)
# for test_str in rep_test_df['sent0']:
#     test_str = re.sub(r1,'',test_str)
#     test_title.append(test_str)
# for test_str in rep_test_df['sent1']:
#     test_str = re.sub(r1,'',test_str)
#     test_content.append(test_str)
#
# train_df['sent0']=train_title
# train_df['sent1']=train_content
# test_df['sent0']=test_title
# test_df['sent1']=test_content
#
# test_df['sent0']=test_df['sent1'].fillna('.')
# train_df['sent1']=train_df['sent1'].fillna('.')
# test_df['sent0']=test_df['sent0'].fillna('.')
# train_df['sent0']=train_df['sent0'].fillna('.')
#
# train_df.to_csv("./replacement/train.csv", index=False)
# test_df.to_csv("./replacement/test.csv", index=False)


import pandas as pd
import re
# 1. Load the dataset into dataframe df
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df = train_df[['id', 'sent0', 'label']]

test_df = test_df[['id', 'sent0']]

# pd.options.display.max_colwidth = 100

# print(df.head(10))
# print('\ntotal entries =' +str(df.shape))

# column_name = (df.columns.values)
# print(column_name)


# #2. creating a copy of original df_cpy for manipulation
# df_cpy = df.copy()
# df_cpy.rename(columns=lambda x: x.strip(), inplace=True)

# # Create an empty datafame df_new for updating with new value
# # df_new = pd.DataFrame(columns=column_name)
# # print(df_new.shape)

# #dropping unnecessary columns and rows
# df_cpy = df_cpy.drop(df_cpy.columns[[0, 1]], axis=1)
# print(df_cpy)
# df_cpy = df_cpy.dropna()

# #drop unnecessary rows:
# #df_cpy = df_cpy.drop(df_cpy[df_cpy.Tag== 'col1_name'].index)
# #df_cpy = df_cpy.drop(df_cpy[df_cpy.Tag == 'col2_name'].index)

# #print('\ntotal entries =' +str(df_cpy.shape))

# # sorting
# df_cpy = df_cpy[df_cpy['issueType']=="Defect"]
# print(df_cpy.head(10))
# print('\ntotal entries =' +str(df_cpy.shape))


# merging 2 cols into new one
# df_cpy['summary_description'] = df_cpy['summary'].astype(str) + df_cpy['description']
# print(df_cpy['summary_description'].head(100))


# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "I'm": "I am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "2nd": "second",
    "NY": "newyork",
    "EU": "Europe",
    "yrs": "years",
    "yoouu": "you",
    "21st": "twenty first",
    "31st": "thirty first",
    "\b1st\b": "first",
    "1st\b": " first",
    "\b4th\b": "fourth",
    "\b5th\b": "fifth",
    "\b6th\b": "sixth",
    "\b7th\b": "seventh",
    "\b8th\b": "eighth",
    "\b9th\b": "ninth",
    "\b13th\b": "thirteenth",
    "\b14th\b": "fourteenth",
    "\b15th\b": "fifteenth",
    "\b16th\b": "sixteenth",
    "\b20th\b": "twentyth",
    "YOOUU": "you",
}

import csv
import string
import wordninja
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)


# clean emojis
def clean_emoji(sen):
    sen = ''.join(c for c in sen if c <= '\uFFFF')
    return sen.replace("  ", " ")


# further cleaning
def clean(sen, remove_stopwords=True, contraction=True, pun=True, lemma_=False):
    sen = re.sub(r'http\S+', 'url', sen, flags=re.MULTILINE)
    sen = re.sub(r'@\S+', '@username', sen)
    sen = re.sub(r'\<a href', ' ', sen)
    sen = re.sub(r'&amp;', '', sen)
    sen = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sen)
    sen = re.sub(r'<br />', ' ', sen)
    sen = re.sub(r"[:()]", "", sen)  # remove ()
    sen = re.sub('\s+$|^\s+', '', sen)  # remove whitespace from start of the line and end of the line
    sen = re.sub(r'[^\x00-\x7f]', r'',
                 sen)  # a single character in the range between  (index 0) and  (index 127) (case sensitive)
    sen = sen.strip(""" '!:?-_().,'"[]{};*""")
    sen = ' '.join([w.strip(""" '!:?-_().,'"[]{};*""") for w in re.split(' ', sen)])

    # sen = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " NUMBER ", sen)

    # spliting words
    string = []
    for x in sen.split():
        if len(x) > 6:
            for i in wordninja.split(x):
                if len(i) > 2:
                    string.append(i)
        else:
            string.append(x)
    sen = " ".join(string)

    contraction
    new_text = []
    for word in sen.split():
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    sen = " ".join(new_text)

    # sen = re.sub(r"[^A-Za-z0-9:(),\'\`]", " ", sen)
    # sen = re.sub(r"\b\d+\b", "", sen)  #remove numbers
    sen = re.sub('\s+', ' ', sen)  # matches any whitespace characte
    sen = re.sub(r'(?:^| )\w(?:$| )', ' ', sen).strip()  # removing single character

    # Optionally, remove stop words
    if remove_stopwords:
        sen = " ".join([i for i in sen.split() if i not in stop])

    # Optionally emove puncuations
    if pun:
        sen = ''.join(ch for ch in sen if ch not in exclude)

    # Optionally lemmatiztion
    if lemma_:
        normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in sen.split())

    return sen.strip().lower()


# Cleaning the dataset

train_sent0 = []
for index, row in train_df['sent0'].iteritems():
    row = clean_emoji(str(row))
    row = clean(row, remove_stopwords=False)
    # print(row)
    train_sent0.append(row)
train_df['sent0'] = train_sent0

# train_sent1 = []
# for index, row in train_df['sent1'].iteritems():
#     row = clean_emoji(str(row))
#     row = clean(row, remove_stopwords=False)
#     # print(row)
#     train_sent1.append(row)
# train_df['sent1'] = train_sent1

test_sent0 = []
for index, row in test_df['sent0'].iteritems():
    row = clean_emoji(str(row))
    row = clean(row, remove_stopwords=False)
    # print(row)
    test_sent0.append(row)
test_df['sent0'] = test_sent0
#
# test_sent1 = []
# for index, row in test_df['sent1'].iteritems():
#     row = clean_emoji(str(row))
#     row = clean(row, remove_stopwords=False)
#     # print(row)
#     test_sent1.append(row)
# test_df['sent1'] = test_sent1

# test_df['sent0'] = test_df['sent1'].fillna('.')
# train_df['sent1'] = train_df['sent1'].fillna('.')
# test_df['sent0'] = test_df['sent0'].fillna('.')
train_df['sent0'] = train_df['sent0'].fillna('.')

train_df.to_csv("./replacement/train.csv", index=False)
test_df.to_csv("./replacement/test.csv", index=False)




#####替换数据划分#########




train_df=pd.read_csv("./replacement/train.csv")
test_df=pd.read_csv("./replacement/test.csv")

train_df = train_df[['id', 'sent0', 'label']]
test_df = test_df[['id', 'sent0']]

train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)
test_df['label']=0

# test_df['sent1']=test_df['sent1'].fillna('.')
# train_df['sent1']=train_df['sent1'].fillna('.')
test_df['sent0']=test_df['sent0'].fillna('.')
train_df['sent0']=train_df['sent0'].fillna('.')

test_df.to_csv('./replacement/test.csv')

import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

X = np.array(train_df.index)
y = train_df.loc[:,'label'].to_numpy()


def generate_data(random_state=42, is_pse_label=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    i = 0
    for train_index, dev_index in skf.split(X, y):
        print(i, "TRAIN:", train_index, "TEST:", dev_index)
        DATA_DIR = "./data_StratifiedKFold_{}/data_replacement_{}/".format(random_state, i)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        tmp_train_df = train_df.iloc[train_index]

        tmp_dev_df = train_df.iloc[dev_index]

        # test_df.to_csv(DATA_DIR + "test.csv")
        if is_pse_label:
            pse_dir = "data_pse_{}/".format(i)
            pse_df = pd.read_csv(pse_dir + 'train.csv')

            tmp_train_df = pd.concat([tmp_train_df, pse_df], ignore_index=True, sort=False)

        tmp_train_df.to_csv(DATA_DIR + "train.csv")
        tmp_dev_df.to_csv(DATA_DIR + "dev.csv")
        print(tmp_train_df.shape, tmp_dev_df.shape)
        i += 1

generate_data(random_state=42, is_pse_label=False)