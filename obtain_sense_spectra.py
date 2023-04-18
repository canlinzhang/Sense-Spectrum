#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:48:53 2023

@author: canlinzhang
"""

import pickle
import nltk
from nltk.corpus import wordnet as wn
import numpy as np

#the class to obtain the spectrum
class ObtainSpectrum:
    
    def __init__(self, path_1, path_2):
        
        with open(path_1, 'rb') as handle_1:
            self.noun_synset_dict = pickle.load(handle_1)
        
        with open(path_2, 'rb') as handle_2:
            self.verb_synset_dict = pickle.load(handle_2)
        
        self.dim = self.noun_synset_dict['entity.n.01'][-1].shape[0]
        
    def return_dict(self):
        
        return(self.noun_synset_dict, self.verb_synset_dict)
        
    def obtain_spectrum_for_word(self, word):
        
        count_noun, count_verb = 0, 0
        
        noun_final_spec = np.zeros((self.dim,), dtype=float)
        verb_final_spec = np.zeros((self.dim,), dtype=float)
    
        for synset in wn.synsets(word):
    
            synset_ = str(synset.name().split(" ")[0])
            
            if synset_ in self.noun_synset_dict:
                
                noun_spec = self.noun_synset_dict[synset_][-1]
                noun_final_spec = np.add(noun_final_spec, noun_spec)
                count_noun += 1
                
            if synset_ in self.verb_synset_dict:
                
                verb_spec = self.verb_synset_dict[synset_][-1]
                verb_final_spec = np.add(verb_final_spec, verb_spec)
                count_verb += 1
                
        #output
        if count_noun > 0:
            out_noun_spec = noun_final_spec/float(count_noun)
        else:
            out_noun_spec = 'no noun synset detected'
            
        if count_verb > 0:
            out_verb_spec = verb_final_spec/float(count_verb)
        else:
            out_verb_spec = 'no verb synset detected'
            
        return(out_noun_spec, out_verb_spec)
    









































