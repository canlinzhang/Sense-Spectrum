# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:16:17 2019

@author: canlinzhang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import csv
import os
import random
import zipfile
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import tensorflow as tf
import nltk
import datetime
import re

from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr

nltk.download('wordnet')

######Hyperparameters####################################
# Training Parameters
learning_rate = 0.0001
display_step = 500
save_step = 5000

noun_batch_size = 600 #better to be divided by 3
verb_batch_size = 300 #better to be divided by 3
SimNet_batch_size = 20

noun_training_step = 1000000
verb_training_step = 500000

#Vocabularies sizes, to be updated by data processing
noun_vocabulary_size = 0
adj_synset_vocabulary_size = 0
adj_true_synset_vocabulary_size = 0
adj_lemma_vocabulary_size = 0
verb_vocabulary_size = 0
adv_synset_vocabulary_size = 0
adv_true_synset_vocabulary_size = 0
adv_lemma_vocabulary_size = 0

#embedding size, fixed
noun_embedding_size = 200
verb_embedding_size = 200

hyper = lambda s: s.hypernyms()
hypo = lambda s: s.hyponyms()



#######################################################
#########Data Processing###############################
#######################################################

########Noun###########################################
#create the noun synset list for tree structure########
#also pick out the words that appeared in the noun synsets
noun_synset_list = []
noun_word = []
noun_synset_dict = {}
synset_index = 0
for synset in list(wn.all_synsets('n')):
    synset_ = str(synset.name().split(" ")[0])
    synset__ = synset_.split(".")
    word_ = synset__[0]
    noun_word.append(word_)
    noun_synset = [] #['synset', [closure]]
    closure = [] #[closure]
    noun_synset.append(synset_)
    noun_synset_dict.update({synset_index:synset_})
    synset_index += 1
    closure.append(synset_)#append the synset itself, otherwise lost information
    for hyper_synset in list(synset.closure(hyper)):
        hyper_synset_ = str(hyper_synset.name().split(" ")[0])
        closure.append(hyper_synset_)
    noun_synset.append(closure)
    noun_synset_list.append(noun_synset)
noun_vocabulary_size += len(noun_synset_list)

#remove the duplicant words
print('removing duplicant noun words')
noun_word = list(dict.fromkeys(noun_word))
print('duplicant noun words removed')

########Verb###########################################
#create the verb synset list for tree structure########
#also pick out the words that appeared in the verb synsets
verb_synset_list = []
verb_word = []
verb_synset_dict = {}
synset_index = 0
for synset in list(wn.all_synsets('v')):
    synset_ = str(synset.name().split(" ")[0])
    synset__ = synset_.split(".")
    word_ = synset__[0]
    verb_word.append(word_)
    verb_synset = [] #['synset', [closure]]
    closure = [] #[closure]
    verb_synset.append(synset_)
    verb_synset_dict.update({synset_index:synset_})
    synset_index += 1
    closure.append(synset_)#append the synset itself, otherwise lost information
    for hyper_synset in list(synset.closure(hyper)):
        hyper_synset_ = str(hyper_synset.name().split(" ")[0])
        closure.append(hyper_synset_)
    verb_synset.append(closure)
    verb_synset_list.append(verb_synset)
verb_vocabulary_size += len(verb_synset_list)

#remove the duplicant words
print('removing duplicant verb words')
verb_word = list(dict.fromkeys(verb_word))
print('duplicant verb words removed')


###########################################################
##############Task Datasets################################
###########################################################

######SimLex_999################################
#read file
with open ('SimLex-999.txt', 'r') as f:
    content_simlex = f.readlines()

content_simlex = [x.strip() for x in content_simlex]

SimLex_999_list = []
SimLex_999_noun_spectrum_pair = []
SimLex_999_verb_spectrum_pair = []
#####take the synsets for each SimLex_999 pair, 
#####then choose the correct synsets for each pair by different similarities
#####also record the initial hypernym_overlapping parameter for each synset combination in a pair

#noun pairs
for i in range(111,777):
    words = content_simlex[i].split()
    score_list = []
    temp_list_1 = []
    temp_list_2 = []
    temp_list_path = []
    temp_list_lch = []
    temp_list_wup = []
    for synset in wn.synsets(words[0], pos=wn.NOUN):
        synset_ = str(synset.name().split(" ")[0])
        temp_list_1.append(synset_)
    for synset in wn.synsets(words[1], pos=wn.NOUN):
        synset_ = str(synset.name().split(" ")[0])
        temp_list_2.append(synset_)
    if (len(temp_list_1) > 0) and (len(temp_list_2) > 0):
        for synset_1 in wn.synsets(words[0], pos=wn.NOUN):
            for synset_2 in wn.synsets(words[1], pos=wn.NOUN):
                temp_score_path = wn.path_similarity(synset_1, synset_2)
                temp_score_lch = wn.lch_similarity(synset_1, synset_2)
                temp_score_wup = wn.wup_similarity(synset_1, synset_2)
                temp_list_path.append(temp_score_path)
                temp_list_lch.append(temp_score_lch)
                temp_list_wup.append(temp_score_wup)
        max_score_path = max(temp_list_path) #pick the correct synset
        max_score_lch = max(temp_list_lch) #pick the correct synset
        max_score_wup = max(temp_list_wup) #pick the correct synset
        score_list.append(i)
        score_list.append(words[0])
        score_list.append(words[1])
        score_list.append(float(words[3]))
        temp_list = []
        for i_1 in range(len(temp_list_1)):
            for i_2 in range(len(temp_list_2)):
                id_1 = noun_synset_dict.keys()[noun_synset_dict.values().index(temp_list_1[i_1])]
                id_2 = noun_synset_dict.keys()[noun_synset_dict.values().index(temp_list_2[i_2])]
                list_1_ = noun_synset_list[id_1][1]
                list_2_ = noun_synset_list[id_2][1]
                cl_ = len(list(set(list_1_).intersection(list_2_)))
                l1_ = len(list_1_)-cl_
                l2_ = len(list_2_)-cl_
                temp_list.append([float(cl_), float(l1_), float(l2_)]) #record initial hyper_overlap param
        score_list.append(temp_list)
        score_list.append(max_score_path)
        score_list.append(max_score_lch)
        score_list.append(max_score_wup)
        SimLex_999_list.append(score_list)
    if i % 10 == 0:
        print('SimLex-999 noun pair', i)

#verb pairs
for i in range(777,999):
    words = content_simlex[i].split()
    score_list = []
    temp_list_1 = []
    temp_list_2 = []
    temp_list_path = []
    temp_list_lch = []
    temp_list_wup = []
    for synset in wn.synsets(words[0], pos=wn.VERB):
        synset_ = str(synset.name().split(" ")[0])
        temp_list_1.append(synset_)
    for synset in wn.synsets(words[1], pos=wn.VERB):
        synset_ = str(synset.name().split(" ")[0])
        temp_list_2.append(synset_)
    if (len(temp_list_1) > 0) and (len(temp_list_2) > 0):
        for synset_1 in wn.synsets(words[0], pos=wn.VERB):
            for synset_2 in wn.synsets(words[1], pos=wn.VERB):
                temp_score_path = wn.path_similarity(synset_1, synset_2)
                temp_score_lch = wn.lch_similarity(synset_1, synset_2)
                temp_score_wup = wn.wup_similarity(synset_1, synset_2)
                temp_list_path.append(temp_score_path)
                temp_list_lch.append(temp_score_lch)
                temp_list_wup.append(temp_score_wup)
        max_score_path = max(temp_list_path) #pick the correct synset
        max_score_lch = max(temp_list_lch) #pick the correct synset
        max_score_wup = max(temp_list_wup) #pick the correct synset
        score_list.append(i)
        score_list.append(words[0])
        score_list.append(words[1])
        score_list.append(float(words[3]))
        temp_list = []
        for i_1 in range(len(temp_list_1)):
            for i_2 in range(len(temp_list_2)):
                id_1 = verb_synset_dict.keys()[verb_synset_dict.values().index(temp_list_1[i_1])]
                id_2 = verb_synset_dict.keys()[verb_synset_dict.values().index(temp_list_2[i_2])]
                list_1_ = verb_synset_list[id_1][1]
                list_2_ = verb_synset_list[id_2][1]
                cl_ = len(list(set(list_1_).intersection(list_2_)))
                l1_ = len(list_1_)-cl_
                l2_ = len(list_2_)-cl_
                temp_list.append([float(cl_), float(l1_), float(l2_)]) #record initial hyper_overlap param       
        score_list.append(temp_list)
        score_list.append(max_score_path)
        score_list.append(max_score_lch)
        score_list.append(max_score_wup)
        SimLex_999_list.append(score_list)
    if i % 10 == 0:
        print('SimLex-999 verb pair', i)


#spearman corr score w.r.t shortest path, lch and wup (three given similarity of WordNet)
spearman_corr_1 = []
spearman_corr_2 = []
for i in range(len(SimLex_999_list)):
    spearman_corr_1.append(SimLex_999_list[i][3])
    spearman_corr_2.append(SimLex_999_list[i][5])
corr_initial_path, p_value_initial = spearmanr(spearman_corr_1, spearman_corr_2)
spearman_corr_1 = []
spearman_corr_2 = []
for i in range(len(SimLex_999_list)):
    spearman_corr_1.append(SimLex_999_list[i][3])
    spearman_corr_2.append(SimLex_999_list[i][6])
corr_initial_lch, p_value_initial = spearmanr(spearman_corr_1, spearman_corr_2)
spearman_corr_1 = []
spearman_corr_2 = []
for i in range(len(SimLex_999_list)):
    spearman_corr_1.append(SimLex_999_list[i][3])
    spearman_corr_2.append(SimLex_999_list[i][7])
corr_initial_wup, p_value_initial = spearmanr(spearman_corr_1, spearman_corr_2)
print('SimLex-999 initial dist', corr_initial_path, corr_initial_lch, corr_initial_wup)



########################################################
#the adjusted formula of Hypernym intersection similarity
spearman_corr_1 = []
spearman_corr_2 = []
for i in range(len(SimLex_999_list)):
    temp_score_list = []
    for j in range(len(SimLex_999_list[i][4])):
        a = SimLex_999_list[i][4][j][0]
        b = SimLex_999_list[i][4][j][1]
        c = SimLex_999_list[i][4][j][2]
        sim_ = a**0.2/(a**0.3+0.5*(b**0.3)+0.5*(c**0.3))
        temp_score_list.append(sim_)
    max_score = max(temp_score_list)
    spearman_corr_1.append(SimLex_999_list[i][3])
    spearman_corr_2.append(max_score)
corr_, p_value_ = spearmanr(spearman_corr_1, spearman_corr_2)
print('SimLex-999 hyper inters sim', corr_)

