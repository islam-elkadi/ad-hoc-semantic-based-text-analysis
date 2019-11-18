import re
import nltk
import json
import pickle

import numpy as np

from os import makedirs
from os.path import exists, join

from statistics import mean

from collections import Counter, defaultdict

from gensim.parsing.preprocessing import *
from gensim.summarization import keywords
from gensim.utils import simple_preprocess
from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import replace_abbreviations, split_sentences
from gensim.models.fasttext import FT
from gensim.models.doc2vec import Doc2Vec as D2V, TaggedDocument
from gensim.models import Word2Vec as WV


from textblob import TextBlob

#-----------------------------------------------
#                 Load/Save Data               
#-----------------------------------------------

def makedir(directory):
    if not exists(directory):
        makedirs(directory)
        return True
    else:
        return False

def load_text_data(directory, read_lines = False):
    with open(directory, "r", encoding = "utf8") as data:
        if read_lines == False:
            return data.read()
        else:
            return data.readlines()

def save_text_data(path, data, mode = "w"):
    with open(path, mode = mode, encoding = "utf-8") as text_file:
        text_file.write(data)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent = 4, ensure_ascii = True)

def save_arrays(name, path = "./"):
    save = os.join(path, name)+".pkl"
    with open(save, 'wb') as outfile:
        pickle.dump(tokens, outfile, pickle.HIGHEST_PROTOCOL)

def load_arrays(name, path = "./"):
    load = os.join(path, name)+".pkl"
    with open(load, 'rb') as infile:
        result = pickle.load(infile)
    return result


#-----------------------------------------------
#                 Clean Text                   
#-----------------------------------------------

def remove_contractions(text):
    """
        Removes contractions to clean sentences
        
        Paras:
            raw: raw text data
        Returns:
            raw: cleaned text
    """
    contractions = { 
                    "ain't": "is not",
                    "aren't": "are not",
                    "can't": "cannot",
                    "could've": "could have",
                    "couldn't": "could not",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he would",
                    "he'll": "he will",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'll": "how will",
                    "how's": "how is",
                    "i'd": "I would",
                    "i'll": "I will",
                    "i'm": "I am",
                    "i've": "I have",
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
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "she'd": "she would",
                    "she'll": "she will",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so as",
                    "that'd": "that would",
                    "that's": "that is",
                    "there'd": "there had",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'll": "they will",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
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
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why has",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "y'all": "you all",
                    "you'd": "you had / you would",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have",
                    "1st": "first",
                    "1 st": "first",
                    "2nd": "second",
                    "2 nd": "second",
                    "3rd":"third",
                    "3 rd":"third",
                    "e\.g\.": "e.g",
                    "i\.e.": "i.e"
                }
    
    for contrac in list(contractions.keys()):
        text = re.sub(contrac, contractions[contrac], text)
    return text

def append_zero_to_decimal(text):
    temp = re.findall(r"\s+\.\d+", text)
    for tmp in temp:
        text = re.sub(tmp, "0{}".format(tmp.strip()), text)
    return text


#-----------------------------------------------
#                   Gensim                   
#-----------------------------------------------

def save_model(model,path):
    model.save(path)
    return True

def load_model(path):
    model=model.load(path)
    return model
