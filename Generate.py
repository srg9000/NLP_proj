import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import sys
import random
import soundex
import time
from metaphone import doublemetaphone
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet as wn
from copy import deepcopy

nltk.download('averaged_perceptron_tagger')

logging.basicConfig(
	 format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def shutdown():
	os.system("shutdown /s /t 1");


WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'
 
def convert(word, from_pos, to_pos):    
	""" Transform words given from/to POS tags """
	 
	synsets = wn.synsets(word, pos=from_pos)
	 
	# Get all lemmas of the word (consider 'a'and 's' equivalent)
	lemmas = [l for s in synsets
				for l in s.lemmas() 
				if str(s.name).split('.')[1] == from_pos
					or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
						and s.name.split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
 
	# Get related forms
	derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]
 
	# filter only the desired pos (consider 'a' and 's' equivalent)
	related_noun_lemmas = [l for drf in derivationally_related_forms
							 for l in drf[1] 
							 if l.synset.name.split('.')[1] == to_pos
								or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
									and l.synset.name.split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
	# Extract the words from the lemmas
	words = [l.name for l in related_noun_lemmas]
	len_words = len(words)
	# Build the result in the form of a list containing tuples (word, probability)
	result = [(w, float(words.count(w))/len_words) for w in set(words)]
	
	result.sort(key=lambda w: -w[1])
	
	result.append(wn.morphy(word,to_pos))
	 
	# return all the possibilities sorted by probability
	return result


def load_corpuses():
		vector_model =  w2v.Word2Vec.load('l222vec800_500feat_0downsample.w2v')
		#itemsets = pd.read_csv('itemset_small_full.csv', names = list('abcdefghijklmnopqrstuvwxyz1'))
		itemsets = open('itemset_small_full.csv','r')
		grammar = open('pos.txt','r')
		
		return [vector_model, grammar, itemsets]
	
	
def search_itemset(itemsets, word):
	similar = set()
	while(1):
			_ = itemsets.readline()
			if len(_) == 0:
				break
			_ = _[0:-1]
			_ = _.split(',')
			# print(_)
			if word in _: 
				[similar.add(x) for x in _ if x not in list('cdet')]
				# print("similar = ",similar)
			if len(similar)>=20:
				print(similar)
				break
		
	itemsets.seek(0,0)
	
	return list(similar)
	

def create_grammar_struct(grammar):
	struct_list = []
	while(1):
			_ = list(grammar.readline().split('_'))
			while(1):
				try:
					_.remove('')
				except:
					break
			if len(_) == 0:
				return struct_list
			
			_[-1] = _[-1][0:-1]
			struct_list.append(_)
#	grammar.seek(0,0)
	return struct_list


def pick_struct(struct_list):
	while True:	
		x = random.choice(struct_list)
		if len(x)>2:
			return x
		
		
def get_similar(vector_model,word):
	return [x[0] for x in vector_model.most_similar(word)]


def wn_pos(treebank_tag):

	if treebank_tag.startswith('J'):
		return wn.ADJ
	elif treebank_tag.startswith('V'):
		return wn.VERB
	elif treebank_tag.startswith('N'):
		return wn.NOUN
	elif treebank_tag.startswith('R'):
		return wn.ADV
	else:
		return ''


def predict_from_itemset(pos, itemset):
	match = []
	for _ in itemset:
		if list(nltk.pos_tag(_))[1] == pos or wn_pos(pos) == wn_pos(list(nltk.pos_tag(_))[1]):
			match.append(_)
			
	return match


def load_tags():
	#DT,PDT,PRP,PP
	with open('DT.pkl','rb') as f:
		dt_list = pickle.load(f)
	with open('PDT.pkl','rb') as f:
		pdt_list = pickle.load(f)
	with open('PRP.pkl','rb') as f:
		prp_list = pickle.load(f)
	with open('PP.pkl','rb') as f:
		pp_list = pickle.load(f)
	
	return [dt_list, pdt_list, prp_list, pp_list]


def compare_rhyme(word1, word2):
	w1 = list(doublemetaphone(word1))
	w2 = list(doublemetaphone(word2))
	
	for i in w1:
		for j in w2:
			if i == j:
				return True
			elif i[len(i)//2:] == j[len(i)//2:]:
				return True
			
	return False


def get_starter(itemsets):
	for i in range(random.randint(1,50)):
		_ = list(itemsets.readline().split(','))
	_[-1] = _[-1][0:-1]
	itemsets.seek(0,0)
#	print(_)
	return _


def remove_slangs(lst):
	l = ['nig','moth', 'fuc']
	ls = deepcopy(lst)
	[ls.remove(x) for x in lst if ('nig' or 'moth' or 'fuc') in x]
	
	return ls


def generate():
	
	previous = ''
	output = []
	
	[dt_list, pdt_list, prp_list, pp_list] = load_tags()
	ext_pos = [dt_list, pdt_list, prp_list, pp_list]
	print("here")		
	[vector_model, grammar, itemsets] = load_corpuses()
	print('x')
	struct_list = create_grammar_struct(grammar)
	print("hereasdasd")
	
	current_set = get_starter(itemsets)
	extras = ['who', 'what','whose', 'where', 'when', 'which', 'in', 'of', 'like']
	while(len(output) < 4):
		cs = set(deepcopy(current_set))
		
		current = []
		
		structure = pick_struct(struct_list)
		print("here")
		
#		for x in cs:
#			print(x, len(cs))
#			current_set.extend(get_similar(vector_model,x))
		#current_set.extend(get_starter(itemsets))	
#		print(structure)
		current_set = list(set(current_set))
#		print("current set = ",current_set)
		try:
			current_set.remove('dont')
		except:
			q=0
		for pos in structure[0:-1]:
			found = 0
#			print(current)
			current_set = list(set(current_set))
#			print(current_set)
			
			for item in current_set:
				if item in current:
					continue
				if str((nltk.pos_tag([item])[0][1]) == str(pos)):
					current.append(item)
#					print("choosing from set")
					found = 1
					current_set.extend(search_itemset(itemsets,item))
					current_set = list(set(current_set))
#					print(found)
					break
				elif (str(wn_pos(nltk.pos_tag([item])[0][1])) == str(wn_pos(pos))):
					current.append(item)
#					print("choosing from set")
					found = 1
					current_set.extend(search_itemset(itemsets,item))
					current_set = list(set(current_set))
#					print(found)
					break
			try:
				[current_set.remove(x) for x in current]
			except:
				q=1
#			print(found)
			current_set = list(set(current_set))
			
			if found == 1:
				continue
			
			else:
			
				for st in ext_pos:
					str(nltk.pos_tag([random.choice(st)])[0][1]) == str(pos)
					current.append(random.choice(st))
					found = 1
					break
				
				if found == 1:
					continue
				
				else:
					current.append(random.choice(extras))
					continue
				
		rh = structure[-1]
		if previous != '':
			for x in current_set:
				if compare_rhyme(previous,x):
					current.append(x)
					break
		previous = current[-1]
		# current = remove_slangs(current)
		print(len(current), len(structure))
		output.append(current)
		cx = deepcopy(current_set)
		# [current_set.pop(current_set.index(ff)) for ff in cx if len(ff)<=2]
		while(len(current_set)>len(cs)//4):
			current_set.pop(random.randint(0,len(current_set)-1))
#		print(output)
			
	
	grammar.close()		
	itemsets.close()
	return [output,struct_list]


[output,grammar] = generate()
print(3*'\n')
for _ in range(len(output)):
	print(*output[_])