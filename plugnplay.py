# Python Code


import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import math
import string
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("en_core_web_sm")
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
buckets = ['Package damaged', 'wrong address','leave order',
       'Delivery company service', 'Delayed delivery', 'credit card',
       'left at front door', 'Package open', 'Customer service',
       'Package left outside', 'Wrong product', 'Delivery person',
       'Delivery date changed', 'order cancelled', 'Dent can','Website', 'Tracking', 'Fast Delivery',
       'Great service', 'Good price', 'Good condition', 'Timely delivery',
       'Easy process','Package not delivered','Package delivered','Poor condition','Poor Service','Arrived Early','Bad experience','Order problem','Order trouble','Items broke','Good packaging','Poor packaging']

def next_nhead(x):
  if x.head == x or x.head.pos_=='NOUN':
    return x.head
  else:
    x = x.head
    return next_nhead(x)
def cmt_pos2_edit(text):
  # print(text) # debug only
  doc2 = nlp(text)
  adj_dict = {}
  cnoun_dict = {}
  verb_dict = {}
  neg_dict = {}
  for tok in doc2:
    #print(tok.lemma_, tok.pos_, tok.dep_, tok.head.lemma_, tok.head.pos_, tok.head.dep_) # debug only
    if tok.pos_ == 'ADJ':
      head = next_nhead(tok)
      if head.pos_ == 'NOUN':
        if head.lemma_ not in adj_dict:
          adj_dict.update({head.lemma_:[tok.lemma_]})
        else:
          adj_dict[head.lemma_].append(tok.lemma_)
      else:
        for child in head.children:
          #print(child,child.pos_,child.lemma_,child.dep_)
          if child.pos_ == 'NOUN' and (child.dep_ == 'nsubj' or child.dep_ == 'nsubjpass'):
            if child.lemma_ not in adj_dict:
              adj_dict.update({child.lemma_:[tok.lemma_]})
            else:
              adj_dict[child.lemma_].append(tok.lemma_)
              
    if tok.pos_ == 'VERB':
      #print('Inside verb',tok)
      for child in tok.children:
        #print(child,child.pos_,child.lemma_,child.dep_)
        if child.pos_ == 'NOUN' and (child.dep_ == 'dobj' or child.dep_ == 'nsubjpass'):
          if child.lemma_ not in verb_dict:
            verb_dict.update({child.lemma_:[tok.lemma_]})
          else:
            verb_dict[child.lemma_].append(tok.lemma_)
        if child.pos_ == 'ADV':
          if child.lemma_ not in verb_dict:
            verb_dict.update({child.lemma_:[tok.lemma_]})
          else:
            verb_dict[child.lemma_].append(tok.lemma_)            
    if tok.pos_ == 'NOUN' and tok.dep_ == 'compound':
      head = tok.head
      if head not in cnoun_dict:
        cnoun_dict.update({head.lemma_:[tok.lemma_]})
      else:
        cnoun_dict[head.lemma_].append(tok.lemma_)
    if tok.dep_ == 'neg':
      head = tok.head
      if head not in neg_dict:
        neg_dict.update({head.lemma_:[tok.lemma_]})
      else:
        cnoun_dict[head.lemma_].append(tok.lemma_)
  l = []
  if len(adj_dict.keys()) > 0:
    for k in adj_dict.keys():
      for v in adj_dict[k]:
        s = str(v)+' '+str(k)
        l.append(s)
  if len(cnoun_dict.keys()) > 0:
    for k in cnoun_dict.keys():
      for v in cnoun_dict[k]:
        s = str(v)+' '+str(k)
        l.append(s)
  if len(verb_dict.keys()) > 0:
    for k in verb_dict.keys():
      for v in verb_dict[k]:
        s = str(v)+' '+str(k)
        l.append(s)
  if len(neg_dict.keys()) > 0:
    for k in neg_dict.keys():
      for v in neg_dict[k]:
        s = str(v)+' '+str(k)
        l.append(s)
  return list(np.unique(l))

def sim_bucketing(ps,model,buckets):
  fin = []
  psble = []
  if len(ps) == 0:
    return fin,psble
  cats = ps
  buck_emb = model.encode(buckets)
  cats_emb = model.encode(cats)
  sim_mat = []
  indx_mat = []

  for i in range(len(cats_emb)):
    row = (cosine_similarity([cats_emb[i]],
      buck_emb)).tolist()[0]
    idx = row.index(max(row))
    idrow = [0]*len(buckets)
    idrow[idx] = 1
    indx_mat.append(idrow)
    sim_mat.append(row)

  emb_df = pd.DataFrame(sim_mat)
  eid_df = pd.DataFrame(indx_mat)
  emb_df.columns = buckets
  emb_df.index = cats
  high_top = list(emb_df.idxmax(axis=1))
  high_val = list(emb_df.max(axis=1))
  for i in range(len(high_val)):
    if high_val[i] >= 0.82:
      if high_top[i] != 'Package delivered':
        fin.append(high_top[i])
    elif high_val[i] >= 0.75:
      if high_top[i] != 'Package delivered':
        psble.append(str(ps[i])+" : "+high_top[i])
  return list(np.unique(fin)),list(np.unique(psble))
