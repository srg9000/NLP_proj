
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re

import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import pyfpgrowth
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys
sys.setrecursionlimit(10**6)


with open("pos_tagged_sentences.txt","rb") as ip:
    transactions = pickle.load(ip)
f = open("itemset_file.csv","w+")
import fp_growth
print("started")
for itemset in fp_growth.find_frequent_itemsets(transactions, 30):
	print(*itemset, file = f, sep = ',', end = '\n')
print("done")
f.close()
ip.close()