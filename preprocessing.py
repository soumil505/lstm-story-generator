# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 20:59:56 2018

@author: soumil
"""


import numpy as np



def get_string_from_file(file_name):
    paragraphs=open(file_name).readlines()
    for i in range (0,len(paragraphs)):
        paragraphs[i]=paragraphs[i].lower().replace('\xa0',' ')
    text=open('train.txt').read().lower().replace('\xa0',' ')
    print('corpus length:', len(text))
    return text

def get_sentences(text,maxlen,step):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('number of sequences:', len(sentences))
    return sentences,next_chars

def vectorize(sentences,next_chars,chars,maxlen):
    print('Vectorization...')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x,y



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    np.seterr(all='raise')
    try:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    except:
        return np.argmax(preds)