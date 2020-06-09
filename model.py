#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:38:15 2020

@author: Zijing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import spacy
import re
import dill


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

stops = dill.load(open('stops.pkd', 'rb'))

df=pd.read_excel('sep.xlsx')



# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])


def stemming_tokenizer(str_input): 
    str_input=str_input.replace('[A-Z]{2,}','')
    doc = nlp(str_input)  
    
    words=[token.lemma_ for token in doc if not token.is_punct]
    words = [re.sub(r"[^A-Za-z@]", "", word) for word in words]    
    words = [re.sub(r"[A-Z]{2,}", "", word) for word in words]
    words = [re.sub(r"\S+\.com", "", word) for word in words]
    words = [re.sub(r"\S+com", "", word) for word in words]
    words = [re.sub(r"\S+@\S+", "", word) for word in words]
    words = [re.sub(r"_", "", word) for word in words]    
    words = [word for word in words if word!=' ']
    words = [word for word in words if len(word)!=0]    
    words = [word for word in words if len(word) < 15]
    words= [re.sub(r'--','', word) for word in words]
    #words=[word.lower() for word in words if word.lower() not in stopwords_lower]
    

    ###get rid of stopwords        
    #string = " ".join(words)
    return words

X_train, X_test, y_train, y_test = train_test_split(df['Body'], 
                                                    df['price'], 
                                                    random_state=0)

news_est = Pipeline([
 ('tvec',TfidfVectorizer(max_df=0.8,stop_words=stops)),
 ('log', LogisticRegression(solver='liblinear'))
])
 

param_grid = {
    'tvec__min_df': [0,40,80],
    #'tvec__max_df': [0.8,0.9,1],
    'tvec__ngram_range': [(1,1),(1,2)],
    'log__C': np.logspace(-4, 4, 4),
    
}

#search = GridSearchCV(news_est, param_grid = param_grid, n_jobs=-1)

news_est.fit(df['Body'],df['price'])


pickle.dump(news_est, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

text = df.iloc[0]['Body']
