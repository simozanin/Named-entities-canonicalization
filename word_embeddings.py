from __future__ import division
from __future__ import print_function

import textdistance
import spacy
import pywikibot
import pickle
from utils import *
from wiki_cleaner import *
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.models import FastText
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from pywikibot.pagegenerators import WikibaseSearchItemPageGenerator
import random
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from collections import defaultdict
from scipy.sparse import csr

from wiki_cleaner import find_interlinks, filter_wiki
import numpy
from numpy import argmax
import numpy as np



import time
import tensorflow as tf

# from gcn.utils import *
from gcn.models import GCN, MLP
###########


site = pywikibot.Site("en", "wikipedia")
repo = site.data_repository()

class Embeddings():
    def __init__(self, file_name="data/original_data/ReVerb/reverb45k_valid.json", ratio = None):
        self.file_name = file_name
        self.ratio = ratio
        self.data, self.sentences = self.load_data()
        self.data_used = []
        self.named_entities = set()
        self.entities_in_triples = set()
        
    def load_data(self, file_name = "data/original_data/ReVerb/reverb45k_valid.json"):
        data = [json.loads(line) for line in open(file_name)]
        sentences = []
        if self.ratio:
            data = random.sample(data, int(self.ratio*len(data)))
        for i in range(len(data)):
            for s in data[i]['src_sentences']:
                sentences.append(clean_sentence(s))
        return data, sentences

    def process_sentences(self):
        
        nlp = spacy.load("en_core_web_sm")  # or any other model
        merge_ents = nlp.create_pipe("merge_entities")
        nlp.add_pipe(merge_ents)


        sentences_tokenized = list()
        named_entities = set()
        for i in range(len(self.data)):
            valid = 0
            sents = []
            new_ents = []
            for s in self.data[i]['src_sentences']:
                new_sent = []
                sent = clean_sentence(s)
                tokens = nlp(sent)
                texts = [t.text  for t in tokens]
                for t in tokens:
                    if t.ent_type:
                        if t.ent_type_ in keep_ents:
                            new_sent.append(t.text.lower())
                            new_ents.append(tuple([t.text.lower(),t.ent_type_]))
                        else:
                            new_sent.append(t.norm_.lower())
                    elif not t.is_punct: #and not t.is_stop and not t.lemma_ == '-pron-':
                        new_sent.append(t.norm_.lower())
                    else:
                        pass
                sents.append(new_sent)
                if str(self.data[i]['triple'][0]) not in texts or self.data[i]['triple'][2] not in texts:
                    valid = 1
                    break
            if valid:
                sentences_tokenized.extend(sents)
                self.data_used.append(self.data[i])
                self.named_entities.update(new_ents)
                self.entities_in_triples.update(set([self.data[i]['triple'][0] ,self.data[i]['triple'][2]]))

    
        self.sentences_tokenized = sentences_tokenized

    def get_entities_from_wiki(self):


        site_links = set()
        # print(self.entities_in_triples)
        for ent in self.entities_in_triples:
            pages = WikibaseSearchItemPageGenerator(ent, language='en', total=10, site=site)

            for p in pages:
                # print("\n\n\n\nYEAH: ",ent)
                try:
                    site_links.add(p.getSitelink(site))
                except:
                    pass
        # print(site_links)

        nlp = spacy.load("en_core_web_sm")  # or any other model
        merge_ents = nlp.create_pipe("merge_entities")
        nlp.add_pipe(merge_ents)
        for link in site_links:
            page = pywikibot.Page(site, link)
            if page.isRedirectPage():
                continue
            text = page.get()
            if '== Notes ==' in text:
                text = text[:text.index('== Notes ==')]
            if '== References ==' in text:
                text = text[:text.index('== References ==')]
            clean_text = filter_wiki(text)
            tokens = nlp(clean_text)
            new_sent = []
            for t in tokens:
                if t.ent_type:
                    if t.ent_type_ in keep_ents:
                        new_sent.append(t.text.lower())
                    else:
                        new_sent.append(t.norm_.lower())
                elif not t.is_punct: #and not t.is_stop and not t.lemma_ == '-pron-':
                    new_sent.append(t.norm_.lower())
                else:
                    pass
            self.sentences_tokenized.append(new_sent)


    def write_tokenized_sents(self, filename=''):
        if not filename:
            filename = self.file_name[self.file_name.rfind("/")+1:-5]
        with open(filename, 'wb') as filehandle:
            pickle.dump(self.sentences_tokenized , filehandle)

    def word_embeddings(self, model = "FastText", filename = 'test_tokenized_s' ):
        if filename:
            with open(filename, 'rb') as filehandle:
                # read the data as binary data stream
                self.sentences_tokenized = pickle.load(filehandle)
                print(len(self.sentences_tokenized))
        if model == "word2vec":       
            path = get_tmpfile("word2vec.model")
            model = Word2Vec(self.sentences_tokenized, size=300, window=30, min_count=1, workers=4, sg=1)
            # model.save("/data/models/"+self.file_name[self.file_name.rfind("/")+1:-5] + ".model")
        elif model == "FastText":
            model = FastText(size=300, window=30, min_count=1)  # instantiate
            model.build_vocab(sentences=self.sentences_tokenized)
            model.train(sentences=self.sentences_tokenized, total_examples=len(self.sentences_tokenized), epochs=10) 
        self.model = model
        self.wv = model.wv


        self.wv.save('fasttext-model.kv')
        print('saved')
   