# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:02:07 2020

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

noun_training_step = 1000000
verb_training_step = 500000

#Vocabularies sizes, to be updated by data processing
noun_vocabulary_size = 0
verb_vocabulary_size = 0

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
    

    ######output the synset##########################
    noun_spectra_ = np.zeros((noun_vocabulary_size, noun_embedding_size))
    for i in range(noun_vocabulary_size):
        holder_synset = []
        holder_synset.append(i)
        _synset_ = sess.run(noun_spectrum, feed_dict={noun_train_inputs_1: holder_synset})
        noun_synset_list[i].append(_synset_)
        noun_spectra_[i,:] = _synset_
        if i % 1000 == 0:
            print('outout noun spectra', i)
    np.savetxt("noun_spectra_.txt", noun_spectra_)

    verb_spectra_ = np.zeros((verb_vocabulary_size, verb_embedding_size))
    for i in range(verb_vocabulary_size):
        holder_synset = []
        holder_synset.append(i)
        _synset_ = sess.run(verb_spectrum, feed_dict={verb_train_inputs_1: holder_synset})
        verb_synset_list[i].append(_synset_)
        verb_spectra_[i,:] = _synset_
        if i % 1000 == 0:
            print('outout verb spectra', i)
    np.savetxt("verb_spectra_.txt", verb_spectra_)


####output the noun and verb synsets  (string)
out_noun = open("noun_synset.txt", "w")
for i in range(noun_vocabulary_size):
    # write each synset to output file
    out_noun.write(noun_synset_list[i][0])
    out_noun.write("\n")
out_noun.close()

out_verb = open("verb_synset.txt", "w")
for i in range(verb_vocabulary_size):
    # write each synset to output file
    out_verb.write(verb_synset_list[i][0])
    out_verb.write("\n")
out_verb.close()


















