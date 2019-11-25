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


##############################################
##############################################
#finding the semantic senses in wordnet of each noun word,
#the training of spectrums will based on it
#We won't pick the word if it only has one synset in WordNet
noun_word_list = []
for i in range(len(noun_word)):
    word_synonyms = []
    temp_synonyms = []
    temp_index = []
    for synset in list(wn.synsets(noun_word[i], pos=wn.NOUN)):
        synset_ = str(synset.name().split(" ")[0])
        temp_synonyms.append(synset_)
        id_ = noun_synset_dict.keys()[noun_synset_dict.values().index(synset_)]
        temp_index.append(id_)
    if len(temp_index) > 1:
        word_synonyms.append(noun_word[i])
        word_synonyms.append(temp_synonyms)
        word_synonyms.append(temp_index)
        noun_word_list.append(word_synonyms)
    if i % 1000 == 0:
        print('finding synonyms for noun', i)

####record the direct hypernyms of each noun synset
for i in range(len(noun_synset_list)):
    synset = wn.synset(noun_synset_list[i][0])
    hypernym_list = []
    hypernym_index = []
    for hypernym in list(synset.hypernyms()):
        hypernym_ = str(hypernym.name().split(" ")[0])
        id_ = noun_synset_dict.keys()[noun_synset_dict.values().index(hypernym_)]
        hypernym_list.append(hypernym_)
        hypernym_index.append(id_)
    noun_synset_list[i].append(hypernym_list)
    noun_synset_list[i].append(hypernym_index)
    if i % 1000 == 0:
        print('direct hypernyms of noun synset', i)


#finding the semantic senses in wordnet of each verb word,
#the training of spectrums will based on it
#We won't pick the word if it only has one synset in WordNet
verb_word_list = []
for i in range(len(verb_word)):
    word_synonyms = []
    temp_synonyms = []
    temp_index = []
    for synset in list(wn.synsets(verb_word[i], pos=wn.VERB)):
        synset_ = str(synset.name().split(" ")[0])
        temp_synonyms.append(synset_)
        id_ = verb_synset_dict.keys()[verb_synset_dict.values().index(synset_)]
        temp_index.append(id_)
    if len(temp_index) > 1:
        word_synonyms.append(verb_word[i])
        word_synonyms.append(temp_synonyms)
        word_synonyms.append(temp_index)
        verb_word_list.append(word_synonyms)
    if i % 1000 == 0:
        print('finding synonyms for verb', i)

####record the direct hypernyms of each verb synset
for i in range(len(verb_synset_list)):
    synset = wn.synset(verb_synset_list[i][0])
    hypernym_list = []
    hypernym_index = []
    for hypernym in list(synset.hypernyms()):
        hypernym_ = str(hypernym.name().split(" ")[0])
        id_ = verb_synset_dict.keys()[verb_synset_dict.values().index(hypernym_)]
        hypernym_list.append(hypernym_)
        hypernym_index.append(id_)
    verb_synset_list[i].append(hypernym_list)
    verb_synset_list[i].append(hypernym_index)
    if i % 1000 == 0:
        print('direct hypernyms of verb synset', i)


###################################################
########Approximated distribution##################

#########noun##################################
noun_neg_sampling_distribution = np.zeros((noun_vocabulary_size), dtype=float)
noun_neg_normalizer = 0.0 #to normalize the NEG distribution to be a valid one

p1 = 1
p2 = 0.6
p3 = 0.3

for i in range(noun_vocabulary_size-1):
    if i < 10000:
        noun_neg_sampling_distribution[i] = p1
        noun_neg_normalizer += p1
    elif i < 30000:
        noun_neg_sampling_distribution[i] = p2
        noun_neg_normalizer += p2
    else:
        noun_neg_sampling_distribution[i] = p3
        noun_neg_normalizer += p3
        
for i in range(noun_vocabulary_size-1):
    noun_neg_sampling_distribution[i] = noun_neg_sampling_distribution[i]/noun_neg_normalizer


#############################################################
####build the label array for output training results#######

#####Noun###########################
label_noun_u = np.zeros((noun_vocabulary_size), dtype=int)
label_noun_v = np.zeros((noun_vocabulary_size), dtype=int)
for i in range(noun_vocabulary_size):
    label_noun_v[i] = i
spectrum_similarity_noun_temp = np.zeros((noun_vocabulary_size), dtype=float)
#####Verb###########################
label_verb_u = np.zeros((verb_vocabulary_size), dtype=int)
label_verb_v = np.zeros((verb_vocabulary_size), dtype=int)
for i in range(verb_vocabulary_size):
    label_verb_v[i] = i
spectrum_similarity_verb_temp = np.zeros((verb_vocabulary_size), dtype=float)


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

path_lch_dist = 0.
for i in range(len(SimLex_999_list)):
    dist = SimLex_999_list[i][5]-SimLex_999_list[i][6]
    path_lch_dist += np.abs(dist)
print('SimLex-999 path lch dist', path_lch_dist)


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


#########################################################
##########Neural Network#################################
#########################################################

########Noun_synset#################################
#create sense embeddings
noun_embeddings = tf.Variable(tf.random_uniform([noun_vocabulary_size, noun_embedding_size], -1.0, 1.0))
#IDs of word 1. shape: [batch_size]
noun_train_inputs_1 = tf.placeholder(tf.int32, shape=[None])
#IDs of word 2. shape: [batch_size]
noun_train_inputs_2 = tf.placeholder(tf.int32, shape=[None])
#label for 1 common with 2: [batch_size]
noun_labels_common = tf.placeholder(tf.float32, shape=[None])
#label for 1/2: [batch_size]
noun_labels_1out2 = tf.placeholder(tf.float32, shape=[None])
#label for 2/1: [batch_size]
noun_labels_2out1 = tf.placeholder(tf.float32, shape=[None])
#get the sense embeddings for 1: [batch_size, embedding_size]
noun_inputs_1 = tf.nn.embedding_lookup(noun_embeddings, noun_train_inputs_1)
#get the sense embeddings for 2: [batch_size, embedding_size]
noun_inputs_2 = tf.nn.embedding_lookup(noun_embeddings, noun_train_inputs_2)
#reshape for outputing spectrum
noun_spectrum = tf.reshape(noun_inputs_1, [-1])

#the common part of embedding 1 and 2. [batch_size, embedding_size]
noun_embd_common = tf.subtract(tf.reduce_min(tf.stack([tf.nn.relu(noun_inputs_1),tf.nn.relu(noun_inputs_2)], axis=2), axis=2),
tf.reduce_min(tf.stack([tf.nn.relu(tf.negative(noun_inputs_1)),tf.nn.relu(tf.negative(noun_inputs_2))], axis=2), axis=2))
#the embedding for 1/2. [batch_size, embedding_size]
noun_embd_1out2 = tf.subtract(tf.nn.relu(tf.subtract(tf.nn.relu(noun_inputs_1), tf.nn.relu(noun_inputs_2))),
tf.nn.relu(tf.subtract(tf.nn.relu(tf.negative(noun_inputs_1)), tf.nn.relu(tf.negative(noun_inputs_2)))))
#the embedding for 2/1. [batch_size, embedding_size]
noun_embd_2out1 = tf.subtract(tf.nn.relu(tf.subtract(tf.nn.relu(noun_inputs_2), tf.nn.relu(noun_inputs_1))),
tf.nn.relu(tf.subtract(tf.nn.relu(tf.negative(noun_inputs_2)), tf.nn.relu(tf.negative(noun_inputs_1)))))
#common score [batch_size]
noun_logits_common = tf.reduce_sum(tf.abs(noun_embd_common), axis=1)
#1/2 score [batch_size]
noun_logits_1out2 = tf.reduce_sum(tf.abs(noun_embd_1out2), axis=1)
#2/1 score [batch_size]
noun_logits_2out1 = tf.reduce_sum(tf.abs(noun_embd_2out1), axis=1)
#compute the error
noun_error = tf.reduce_mean(tf.concat([tf.abs(tf.subtract(noun_labels_common, noun_logits_common)),
                   tf.abs(tf.subtract(noun_labels_1out2, noun_logits_1out2)),
tf.abs(tf.subtract(noun_labels_2out1, noun_logits_2out1))], 0))
#training step
noun_train_step = tf.train.AdamOptimizer(learning_rate).minimize(noun_error)
#output the result, [batch_size]
noun_matching_score = tf.subtract(noun_logits_common, tf.add(noun_logits_1out2, noun_logits_2out1))
noun_matching_score_div = tf.div(noun_logits_common, tf.add_n([noun_logits_common, noun_logits_1out2, noun_logits_2out1]))


########Verb_synset#################################
#create sense embeddings
verb_embeddings = tf.Variable(tf.random_uniform([verb_vocabulary_size, verb_embedding_size], -1.0, 1.0))
#IDs of word 1. shape: [batch_size]
verb_train_inputs_1 = tf.placeholder(tf.int32, shape=[None])
#IDs of word 2. shape: [batch_size]
verb_train_inputs_2 = tf.placeholder(tf.int32, shape=[None])
#label for 1 common with 2: [batch_size]
verb_labels_common = tf.placeholder(tf.float32, shape=[None])
#label for 1/2: [batch_size]
verb_labels_1out2 = tf.placeholder(tf.float32, shape=[None])
#label for 2/1: [batch_size]
verb_labels_2out1 = tf.placeholder(tf.float32, shape=[None])
#get the sense embeddings for 1: [batch_size, embedding_size]
verb_inputs_1 = tf.nn.embedding_lookup(verb_embeddings, verb_train_inputs_1)
#get the sense embeddings for 2: [batch_size, embedding_size]
verb_inputs_2 = tf.nn.embedding_lookup(verb_embeddings, verb_train_inputs_2)
#reshape for outputing spectrum
verb_spectrum = tf.reshape(verb_inputs_1, [-1])

#the common part of embedding 1 and 2. [batch_size, embedding_size]
verb_embd_common = tf.subtract(tf.reduce_min(tf.stack([tf.nn.relu(verb_inputs_1),tf.nn.relu(verb_inputs_2)], axis=2), axis=2),
tf.reduce_min(tf.stack([tf.nn.relu(tf.negative(verb_inputs_1)),tf.nn.relu(tf.negative(verb_inputs_2))], axis=2), axis=2))
#the embedding for 1/2. [batch_size, embedding_size]
verb_embd_1out2 = tf.subtract(tf.nn.relu(tf.subtract(tf.nn.relu(verb_inputs_1), tf.nn.relu(verb_inputs_2))),
tf.nn.relu(tf.subtract(tf.nn.relu(tf.negative(verb_inputs_1)), tf.nn.relu(tf.negative(verb_inputs_2)))))
#the embedding for 2/1. [batch_size, embedding_size]
verb_embd_2out1 = tf.subtract(tf.nn.relu(tf.subtract(tf.nn.relu(verb_inputs_2), tf.nn.relu(verb_inputs_1))),
tf.nn.relu(tf.subtract(tf.nn.relu(tf.negative(verb_inputs_2)), tf.nn.relu(tf.negative(verb_inputs_1)))))
#common score [batch_size]
verb_logits_common = tf.reduce_sum(tf.abs(verb_embd_common), axis=1)
#1/2 score [batch_size]
verb_logits_1out2 = tf.reduce_sum(tf.abs(verb_embd_1out2), axis=1)
#2/1 score [batch_size]
verb_logits_2out1 = tf.reduce_sum(tf.abs(verb_embd_2out1), axis=1)
#compute the error
verb_error = tf.reduce_mean(tf.concat([tf.abs(tf.subtract(verb_labels_common, verb_logits_common)),
                   tf.abs(tf.subtract(verb_labels_1out2, verb_logits_1out2)),
tf.abs(tf.subtract(verb_labels_2out1, verb_logits_2out1))], 0))
#training step
verb_train_step = tf.train.AdamOptimizer(learning_rate).minimize(verb_error)
#output the result, [batch_size]
verb_matching_score = tf.subtract(verb_logits_common, tf.add(verb_logits_1out2, verb_logits_2out1))
verb_matching_score_div = tf.div(verb_logits_common, tf.add_n([verb_logits_common, verb_logits_1out2, verb_logits_2out1]))



#######################################################
###########Run the network#############################
#######################################################
saver = tf.train.Saver()
init = tf.global_variables_initializer()
conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=conf) as sess:    
    # Run the initializer.
    sess.run(init)
    #saver.restore(sess, 'checkpoints/sense_spectrum/model.ckpt')
    out_file = open("sense_spectrum_error", "wb")
    log_writer = csv.writer(out_file, delimiter='\t', quotechar='|',  quoting=csv.QUOTE_MINIMAL)
    out_file_noun = open("sense_spectrum_noun", "wb")
    log_writer_noun = csv.writer(out_file_noun, delimiter='\t', quotechar='|',  quoting=csv.QUOTE_MINIMAL)
    out_file_verb = open("sense_spectrum_verb", "wb")
    log_writer_verb = csv.writer(out_file_verb, delimiter='\t', quotechar='|',  quoting=csv.QUOTE_MINIMAL)
    out_file_pair = open("sense_spectrum_pair", "wb")
    log_writer_pair = csv.writer(out_file_pair, delimiter='\t', quotechar='|',  quoting=csv.QUOTE_MINIMAL)
    out_file_bin = open("sense_spectrum_bin", "wb")
    log_writer_bin = csv.writer(out_file_bin, delimiter='\t', quotechar='|',  quoting=csv.QUOTE_MINIMAL)
 
    ################################################
    #########start training#########################

    #########Noun_training####################
    step = 0
    for step in range(noun_training_step):
        holder_inputs_1 = []
        holder_inputs_2 = []
        holder_labels_common = []
        holder_labels_1out2 = []
        holder_labels_2out1 = []
        
        holder_1 = np.random.choice(np.arange(0, noun_vocabulary_size), size=noun_batch_size, p=noun_neg_sampling_distribution)
        holder_2 = np.random.choice(np.arange(0, noun_vocabulary_size), size=noun_batch_size, p=noun_neg_sampling_distribution)
        holder_word = np.random.choice(np.arange(0, len(noun_word_list)), size=noun_batch_size)
        
        for i in range(noun_batch_size):
            if i % 3 == 0: #find synonyms from noun_word_list
                random_synonyms = random.sample(noun_word_list[holder_word[i]][2], 2)
                holder_inputs_1.append(random_synonyms[0])
                holder_inputs_2.append(random_synonyms[1])
                list_1 = noun_synset_list[holder_inputs_1[-1]][1]
                list_2 = noun_synset_list[holder_inputs_2[-1]][1]
            if i % 3 == 1: #find hypernyms from noun_synset_list
                holder_inputs_1.append(holder_1[i])
                temp_hyper_holder = noun_synset_list[holder_inputs_1[-1]][3]
                if len(temp_hyper_holder) >= 1:
                    random_hypernym = random.sample(temp_hyper_holder, 1)
                    holder_inputs_2.append(random_hypernym[0])
                else:
                    holder_inputs_2.append(holder_2[i])
                list_1 = noun_synset_list[holder_inputs_1[-1]][1]
                list_2 = noun_synset_list[holder_inputs_2[-1]][1]
            if i % 3 == 2: #random synsets as negative samples
                holder_inputs_1.append(holder_1[i])
                holder_inputs_2.append(holder_2[i])
                list_1 = noun_synset_list[holder_inputs_1[-1]][1]
                list_2 = noun_synset_list[holder_inputs_2[-1]][1]
            common_length = len(list(set(list_1).intersection(list_2)))
            holder_labels_common.append(common_length)
            holder_labels_1out2.append(len(list_1)-common_length)
            holder_labels_2out1.append(len(list_2)-common_length)
            del(list_1, list_2)
        del(holder_1, holder_2)
        sess.run(noun_train_step, feed_dict={noun_train_inputs_1: holder_inputs_1,
                                        noun_train_inputs_2: holder_inputs_2,
                                        noun_labels_common: holder_labels_common,
                                        noun_labels_1out2: holder_labels_1out2,
                                        noun_labels_2out1: holder_labels_2out1})
        
        if step % display_step == 0:
            noun_error_ = sess.run(noun_error, feed_dict={noun_train_inputs_1: holder_inputs_1,
                                        noun_train_inputs_2: holder_inputs_2,
                                        noun_labels_common: holder_labels_common,
                                        noun_labels_1out2: holder_labels_1out2,
                                        noun_labels_2out1: holder_labels_2out1})
            print('noun_training', step, noun_error_)
            log_writer.writerow(['noun_training', step, noun_error_])

        if step % save_step == 0:
            saver_path = saver.save(sess, 'checkpoints/sense_spectrum/model.ckpt') 
            print('checkpoint saved')


    #########verb_training####################
    step = 0
    for step in range(verb_training_step):
        holder_inputs_1 = []
        holder_inputs_2 = []
        holder_labels_common = []
        holder_labels_1out2 = []
        holder_labels_2out1 = []
        
        holder_1 = np.random.choice(np.arange(0, verb_vocabulary_size), size=verb_batch_size)
        holder_2 = np.random.choice(np.arange(0, verb_vocabulary_size), size=verb_batch_size)
        holder_word = np.random.choice(np.arange(0, len(verb_word_list)), size=verb_batch_size)
        
        for i in range(verb_batch_size):
            if i % 3 == 0: #find synonyms from verb_word_list
                random_synonyms = random.sample(verb_word_list[holder_word[i]][2], 2)
                holder_inputs_1.append(random_synonyms[0])
                holder_inputs_2.append(random_synonyms[1])
                list_1 = verb_synset_list[holder_inputs_1[-1]][1]
                list_2 = verb_synset_list[holder_inputs_2[-1]][1]
            if i % 3 == 1: #find hypernyms from verb_synset_list
                holder_inputs_1.append(holder_1[i])
                temp_hyper_holder = verb_synset_list[holder_inputs_1[-1]][3]
                if len(temp_hyper_holder) >= 1:
                    random_hypernym = random.sample(temp_hyper_holder, 1)
                    holder_inputs_2.append(random_hypernym[0])
                else:
                    holder_inputs_2.append(holder_2[i])
                list_1 = verb_synset_list[holder_inputs_1[-1]][1]
                list_2 = verb_synset_list[holder_inputs_2[-1]][1]
            if i % 3 == 2: #random synsets as negative samples
                holder_inputs_1.append(holder_1[i])
                holder_inputs_2.append(holder_2[i])
                list_1 = verb_synset_list[holder_inputs_1[-1]][1]
                list_2 = verb_synset_list[holder_inputs_2[-1]][1]
            common_length = len(list(set(list_1).intersection(list_2)))
            holder_labels_common.append(common_length)
            holder_labels_1out2.append(len(list_1)-common_length)
            holder_labels_2out1.append(len(list_2)-common_length)
            del(list_1, list_2)
        del(holder_1, holder_2)
        sess.run(verb_train_step, feed_dict={verb_train_inputs_1: holder_inputs_1,
                                        verb_train_inputs_2: holder_inputs_2,
                                        verb_labels_common: holder_labels_common,
                                        verb_labels_1out2: holder_labels_1out2,
                                        verb_labels_2out1: holder_labels_2out1})
        
        if step % display_step == 0:
            verb_error_ = sess.run(verb_error, feed_dict={verb_train_inputs_1: holder_inputs_1,
                                        verb_train_inputs_2: holder_inputs_2,
                                        verb_labels_common: holder_labels_common,
                                        verb_labels_1out2: holder_labels_1out2,
                                        verb_labels_2out1: holder_labels_2out1})
            print('verb_training', step, verb_error_)
            log_writer.writerow(['verb_training', step, verb_error_])

        if step % save_step == 0:
            saver_path = saver.save(sess, 'checkpoints/sense_spectrum/model.ckpt') 
            print('checkpoint saved')


    ###################################################
    ####spectrum performance with SimLex-999###########
    ###################################################

    ####pick out corresponding spectrums for each SimLex_999 pair
    ####we use the overlapping score on spectrums, 
    ####then take the pair with maximum score
    SimLex_999_list_spectrum = []
    
    ######noun pair###################################    
    SimLex_999_noun_list = []
    valid_noun_pair = 0
    for i in range(111,777):
        words = content_simlex[i].split()
        score_list = []
        temp_list_1 = []
        temp_list_2 = []
        for synset in wn.synsets(words[0], pos=wn.NOUN):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_1.append(synset_)
        for synset in wn.synsets(words[1], pos=wn.NOUN):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_2.append(synset_)
        if (len(temp_list_1) > 0) and (len(temp_list_2) > 0):
            score_list.append(valid_noun_pair)
            score_list.append(words[0])
            score_list.append(words[1])
            score_list.append(float(words[3]))
            holder_1 = []
            holder_2 = []
            chosen_holder_1 = []
            chosen_holder_2 = []
            for i_1 in range(len(temp_list_1)):
                for i_2 in range(len(temp_list_2)):
                    id_1 = noun_synset_dict.keys()[noun_synset_dict.values().index(temp_list_1[i_1])]
                    id_2 = noun_synset_dict.keys()[noun_synset_dict.values().index(temp_list_2[i_2])]
                    holder_1.append(id_1)
                    holder_2.append(id_2)
            noun_matching_score_div_ = sess.run(noun_matching_score_div, feed_dict={noun_train_inputs_1: holder_1,
                                        noun_train_inputs_2: holder_2})
            max_overlapping_position = int(np.argmax(noun_matching_score_div_))
            chosen_id_1 = holder_1[max_overlapping_position]
            chosen_id_2 = holder_2[max_overlapping_position]
            max_pair_score = max(noun_matching_score_div_)
            chosen_holder_1.append(chosen_id_1)
            chosen_holder_2.append(chosen_id_2)
            chosen_spectrum_1 = sess.run(noun_spectrum, feed_dict={noun_train_inputs_1: chosen_holder_1})
            chosen_spectrum_2 = sess.run(noun_spectrum, feed_dict={noun_train_inputs_1: chosen_holder_2})
            score_list.append(chosen_spectrum_1)
            score_list.append(chosen_spectrum_2)
            score_list.append(max_pair_score)
            SimLex_999_noun_list.append(score_list)
            SimLex_999_list_spectrum.append(score_list)
            #store spectrums in files, easier when training shallow network
            with open("SimLex_999_spectrum/noun/" + str(valid_noun_pair) + "_" + str(1), 'w') as f_1:
                for item in chosen_spectrum_1:
                    f_1.write("%s\n" % item)
            with open("SimLex_999_spectrum/noun/" + str(valid_noun_pair) + "_" + str(2), 'w') as f_2:
                for item in chosen_spectrum_2:
                    f_2.write("%s\n" % item)
            valid_noun_pair += 1
        if i % 10 == 0:
            print('outputting noun pair spectrums', i)
    #label file
    with open("SimLex_999_spectrum/noun/noun_label", 'w') as f_n:
        for l in range(len(SimLex_999_noun_list)):
            f_n.write("%s\n" % SimLex_999_noun_list[l][3])

    ######verb pair################################    
    SimLex_999_verb_list = []
    valid_verb_pair = 0
    for i in range(777,999):
        words = content_simlex[i].split()
        score_list = []
        temp_list_1 = []
        temp_list_2 = []
        for synset in wn.synsets(words[0], pos=wn.VERB):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_1.append(synset_)
        for synset in wn.synsets(words[1], pos=wn.VERB):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_2.append(synset_)
        if (len(temp_list_1) > 0) and (len(temp_list_2) > 0):
            score_list.append(valid_verb_pair)
            score_list.append(words[0])
            score_list.append(words[1])
            score_list.append(float(words[3]))
            holder_1 = []
            holder_2 = []
            chosen_holder_1 = []
            chosen_holder_2 = []
            for i_1 in range(len(temp_list_1)):
                for i_2 in range(len(temp_list_2)):
                    id_1 = verb_synset_dict.keys()[verb_synset_dict.values().index(temp_list_1[i_1])]
                    id_2 = verb_synset_dict.keys()[verb_synset_dict.values().index(temp_list_2[i_2])]
                    holder_1.append(id_1)
                    holder_2.append(id_2)
            verb_matching_score_div_ = sess.run(verb_matching_score_div, feed_dict={verb_train_inputs_1: holder_1,
                                        verb_train_inputs_2: holder_2})
            max_overlapping_position = int(np.argmax(verb_matching_score_div_))
            chosen_id_1 = holder_1[max_overlapping_position]
            chosen_id_2 = holder_2[max_overlapping_position]
            max_pair_score = max(verb_matching_score_div_)          
            chosen_holder_1.append(chosen_id_1)
            chosen_holder_2.append(chosen_id_2)
            chosen_spectrum_1 = sess.run(verb_spectrum, feed_dict={verb_train_inputs_1: chosen_holder_1})
            chosen_spectrum_2 = sess.run(verb_spectrum, feed_dict={verb_train_inputs_1: chosen_holder_2})
            score_list.append(chosen_spectrum_1)
            score_list.append(chosen_spectrum_2)
            score_list.append(max_pair_score)
            SimLex_999_verb_list.append(score_list)
            SimLex_999_list_spectrum.append(score_list)
            #store spectrums in files, easier when training shallow network
            with open("SimLex_999_spectrum/verb/" + str(valid_verb_pair) + "_" + str(1), 'w') as f_1:
                for item in chosen_spectrum_1:
                    f_1.write("%s\n" % item)
            with open("SimLex_999_spectrum/verb/" + str(valid_verb_pair) + "_" + str(2), 'w') as f_2:
                for item in chosen_spectrum_2:
                    f_2.write("%s\n" % item)
            valid_verb_pair += 1
        if i % 10 == 0:
            print('outputting verb pair spectrums', i)
    #label file
    with open("SimLex_999_spectrum/verb/verb_label", 'w') as f_v:
        for l in range(len(SimLex_999_verb_list)):
            f_v.write("%s\n" % SimLex_999_verb_list[l][3])


    ###################################################
    ###compute the HIS similarity between SimLex-999 synset pairs
    synset_pair_list = []
    synset_bin_list = []
    synset_bin_score = []
    for i in range(111,777):#noun pairs
        synset_temp_list = []
        synset_temp_score = []
        words = content_simlex[i].split()
        temp_list_1 = []
        temp_list_2 = []
        for synset in wn.synsets(words[0], pos=wn.NOUN):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_1.append(synset_)
        for synset in wn.synsets(words[1], pos=wn.NOUN):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_2.append(synset_)
        if (len(temp_list_1) > 0) and (len(temp_list_2) > 0):
            for j in range(len(temp_list_1)):
                for k in range(len(temp_list_2)):
                    holder_synset_1 = []
                    holder_synset_2 = []
                    id_1 = noun_synset_dict.keys()[noun_synset_dict.values().index(temp_list_1[j])]
                    id_2 = noun_synset_dict.keys()[noun_synset_dict.values().index(temp_list_2[k])]
                    list_1_ = noun_synset_list[id_1][1]
                    list_2_ = noun_synset_list[id_2][1]
                    cl_ = len(list(set(list_1_).intersection(list_2_)))
                    l1_ = len(list_1_)-cl_
                    l2_ = len(list_2_)-cl_
                    holder_synset_1.append(id_1)
                    holder_synset_2.append(id_2)
                    noun_logits_common_ = sess.run(noun_logits_common, 
                                feed_dict={noun_train_inputs_1: holder_synset_1,
                                        noun_train_inputs_2: holder_synset_2})
                    noun_logits_1out2_ = sess.run(noun_logits_1out2, 
                                feed_dict={noun_train_inputs_1: holder_synset_1,
                                        noun_train_inputs_2: holder_synset_2})
                    noun_logits_2out1_ = sess.run(noun_logits_2out1, 
                                feed_dict={noun_train_inputs_1: holder_synset_1,
                                        noun_train_inputs_2: holder_synset_2})
                    synset_temp_list.append([temp_list_1[j], temp_list_2[k],
                                cl_,l1_,l2_,noun_logits_common_,
                                noun_logits_1out2_,noun_logits_2out1_])
                    temp_score = cl_**0.25/(cl_**0.25+0.5*(l1_**0.25)+0.5*(l2_**0.25))
                    synset_temp_score.append(temp_score)
                    log_writer_bin.writerow([temp_list_1[j], temp_list_2[k],
                                cl_,l1_,l2_,noun_logits_common_,
                                noun_logits_1out2_,noun_logits_2out1_])
            synset_bin_list.append(synset_temp_list)
            synset_bin_score.append(synset_temp_score)
        if i % 10 == 0:
            print('initial HIS score on noun pair', i)
        
    for i in range(777,999):#verb pairs
        synset_temp_list = []
        synset_temp_score = []
        words = content_simlex[i].split()
        temp_list_1 = []
        temp_list_2 = []
        for synset in wn.synsets(words[0], pos=wn.VERB):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_1.append(synset_)
        for synset in wn.synsets(words[1], pos=wn.VERB):
            synset_ = str(synset.name().split(" ")[0])
            temp_list_2.append(synset_)
        if (len(temp_list_1) > 0) and (len(temp_list_2) > 0):
            for j in range(len(temp_list_1)):
                for k in range(len(temp_list_2)):
                    holder_synset_1 = []
                    holder_synset_2 = []
                    id_1 = verb_synset_dict.keys()[verb_synset_dict.values().index(temp_list_1[j])]
                    id_2 = verb_synset_dict.keys()[verb_synset_dict.values().index(temp_list_2[k])]
                    list_1_ = verb_synset_list[id_1][1]
                    list_2_ = verb_synset_list[id_2][1]
                    cl_ = len(list(set(list_1_).intersection(list_2_)))
                    l1_ = len(list_1_)-cl_
                    l2_ = len(list_2_)-cl_
                    holder_synset_1.append(id_1)
                    holder_synset_2.append(id_2)
                    verb_logits_common_ = sess.run(verb_logits_common, 
                                feed_dict={verb_train_inputs_1: holder_synset_1,
                                        verb_train_inputs_2: holder_synset_2})
                    verb_logits_1out2_ = sess.run(verb_logits_1out2, 
                                feed_dict={verb_train_inputs_1: holder_synset_1,
                                        verb_train_inputs_2: holder_synset_2})
                    verb_logits_2out1_ = sess.run(verb_logits_2out1, 
                                feed_dict={verb_train_inputs_1: holder_synset_1,
                                        verb_train_inputs_2: holder_synset_2})
                    synset_temp_list.append([temp_list_1[j], temp_list_2[k],
                                cl_,l1_,l2_,verb_logits_common_,
                                verb_logits_1out2_,verb_logits_2out1_,temp_score])
                    temp_score = cl_**0.25/(cl_**0.25+0.5*(l1_**0.25)+0.5*(l2_**0.25))
                    synset_temp_score.append(temp_score)
                    log_writer_bin.writerow([temp_list_1[j], temp_list_2[k],
                                cl_,l1_,l2_,verb_logits_common_,
                                verb_logits_1out2_,verb_logits_2out1_,temp_score])
            synset_bin_list.append(synset_temp_list)
            synset_bin_score.append(synset_temp_score)                    
        if i % 10 == 0:
            print('initial HIS score on verb pair', i)


    #record the SimLex-999 most matching pairs
    for i in range(len(synset_bin_score)):
        max_score_pair = int(np.argmax(synset_bin_score[i]))
        synset_pair_list.append(synset_bin_list[i][max_score_pair])
        log_writer_pair.writerow([synset_bin_list[i][max_score_pair]])


    ######output the synset##########################
    for i in range(noun_vocabulary_size):
        holder_synset = []
        holder_synset.append(i)
        _synset_ = sess.run(noun_spectrum, feed_dict={noun_train_inputs_1: holder_synset})
        noun_synset_list[i].append(_synset_)

    for i in range(verb_vocabulary_size):
        holder_synset = []
        holder_synset.append(i)
        _synset_ = sess.run(verb_spectrum, feed_dict={verb_train_inputs_1: holder_synset})
        verb_synset_list[i].append(_synset_)

    ##########################################
    #compute the noun spectrum similarity
    for i in range(noun_vocabulary_size):
        label_noun_u[:] = i
        noun_matching_score_ = sess.run(noun_matching_score, feed_dict={noun_train_inputs_1: label_noun_u,
                                        noun_train_inputs_2: label_noun_v})
        spectrum_similarity_noun_temp[:] = noun_matching_score_
        spectrum_similarity_noun_temp[i] = (-1)*10000000
        i_list = [] #[[...synset_1...], [...synset_2...], ...]
        i_list.append(noun_synset_list[i][0])
        i_list.append('   :   ')        
        for j in range(5):
            closest_word_list = []
            closest_word = np.argmax(spectrum_similarity_noun_temp[:])
            closest_word_list.append(closest_word) #lemma ID      
            closest_word_list.append(noun_synset_list[closest_word][0]) #lemma
            closest_word_list.append(spectrum_similarity_noun_temp[closest_word]) #score
            i_list.append(closest_word_list)
            spectrum_similarity_noun_temp[closest_word] = (-1)*10000000
        noun_synset_list[i].append(i_list)
        log_writer_noun.writerow([i_list])
        if i % 100 == 0:
            print('noun similarity list reached', i)

    ##########################################
    #compute the verb spectrum similarity
    for i in range(verb_vocabulary_size):
        label_verb_u[:] = i
        verb_matching_score_ = sess.run(verb_matching_score, feed_dict={verb_train_inputs_1: label_verb_u,
                                        verb_train_inputs_2: label_verb_v})
        spectrum_similarity_verb_temp[:] = verb_matching_score_
        spectrum_similarity_verb_temp[i] = (-1)*10000000
        i_list = [] #[[...synset_1...], [...synset_2...], ...]
        i_list.append(verb_synset_list[i][0])
        i_list.append('   :   ')        
        for j in range(5):
            closest_word_list = []
            closest_word = np.argmax(spectrum_similarity_verb_temp[:])
            closest_word_list.append(closest_word) #lemma ID      
            closest_word_list.append(verb_synset_list[closest_word][0]) #lemma
            closest_word_list.append(spectrum_similarity_verb_temp[closest_word]) #score
            i_list.append(closest_word_list)
            spectrum_similarity_verb_temp[closest_word] = (-1)*10000000
        verb_synset_list[i].append(i_list)
        log_writer_verb.writerow([i_list])
        if i % 100 == 0:
            print('verb similarity list reached', i)

   

