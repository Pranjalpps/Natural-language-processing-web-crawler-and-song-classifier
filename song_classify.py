#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:59:19 2020

@author: pranjal
"""


import os
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
import matplotlib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection,preprocessing,naive_bayes,metrics



def get_wordnet_pos(word):
    # Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)    #returns pos value that can be accepted by lemmatize function



def preprocess(text):
    
    # sent tokenize----------------------------------------------
    text = text.strip()    #removing extra white spaces
    s_tokenized = sent_tokenize(text)
    
    #cleaning up for word tokenizing-----------------------------
    import re
    text = re.sub(r'\d+', '', text)  #removing numbers
    import string
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) #removing symbols
    text=text.strip()
    text = re.sub('\s+', ' ', text)  # remove newline chars
    
    #removing stop words ----------------------------------------
    filtered_text=[]
    stop_words=set(stopwords.words("english"))
    w_tokenized = word_tokenize((text))
    for w in w_tokenized:
        if w not in stop_words:
            filtered_text.append(w)
            
    #lemmatizing  ----------------------------------------------
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(w)) for word in filtered_text]
    
    return (" ".join(lemmas))      #returning string of lemmas which are to be used for clustering





def train_model(classifier, feature_vector_train, label, feature_vector_valid,valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return (metrics.accuracy_score(predictions, valid_y))

def main():
    
    data=[]
    labels=[]
    Path = r"/home/pranjal/Desktop/song_classification" #Path to the data set downloaded by the web crawler
    filelist = os.listdir(Path)
    for i in range(4):      #taking 1 file each from any 3 categories for text preprocessing
        for j in range(len(os.listdir(Path+"/"+filelist[i]))):
            
            new_path=Path+"/"+filelist[i]+"/"+os.listdir(Path+"/"+filelist[i])[j]
            file = open(new_path, 'r')
            text = file.read()
            text=text.strip()
            file.close()
            text=preprocess(text)
            data.append(text)
            labels.append(filelist[i])
            
    df=pd.DataFrame()        
    df["songs"]=data
    df["labels"]=labels
    
    df['songs'].replace('', np.nan, inplace=True)
    df.dropna(subset=['songs'], inplace=True)
    
    from sklearn.utils import shuffle
    df = shuffle(df)              #shuffling dataframe before splitting for train /test data set
    df.reset_index(inplace=True, drop=True)    #resetting the indexes shuffled due to above operation
    train_df=df
    
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['songs'], train_df['labels'])
    
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    
    # tf-idf bag of words model
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train_df['songs'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    
    
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf,valid_y)
    print ("Accuracy for Naive Bayes classifier is : ", accuracy)
    



if __name__=="__main__":
    main()
