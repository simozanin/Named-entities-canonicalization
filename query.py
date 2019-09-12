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


def query():
    site = pywikibot.Site("en", "wikipedia")
    repo = site.data_repository()

    data = [json.loads(line) for line in open("data/original_data/ReVerb/reverb45k_valid.json")]

    named_entities = set()
    entities_in_triples = set()
    named_entities = set()

    neighbors_per_triple = defaultdict()
    wv = KeyedVectors.load("fasttext-model.kv", mmap='r')

    nlp = spacy.load("en_core_web_sm")  # or any other model
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    
    correct_subs = 29
    correct_obj = 50
    correct_both = 21
    g_correct_subs = 14
    g_correct_obj = 17
    g_correct_both = 6
    tot = 78

    for j in range(2,500,3):

        data_item = data[j]
        tID = data_item['_id']
        # if tID != 4581:
        #     continue
        t_neighbors = defaultdict(set)
        valid = 0
        named_entities = set()
        entities_in_triples = set()
        named_entities = set()
        new_ents = set()
        for s in data_item['src_sentences']:
            sent = clean_sentence(s)
            tokens = nlp(sent)
            texts = [t.text  for t in tokens]
            for t in tokens:
                if t.ent_type:
                    if t.ent_type_ in keep_ents and t.text.lower() in wv:
                        new_ents.add(t.text.lower())

            if data_item['triple'][0].lower() not in texts or data_item['triple'][2].lower() not in texts:
                valid = 1

        if valid and data_item['triple'][0].lower() in wv and data_item['triple'][2].lower() in wv:
            named_entities.update(new_ents)
            entities_in_triples.update(set([data_item['triple'][0] ,data_item['triple'][2]]))
            tx = [wv.word_vec(data_item['triple'][0].lower()), wv.word_vec(data_item['triple'][2].lower())]
            for ent in new_ents:
                if ent in [data_item['triple'][0].lower() ,data_item['triple'][2].lower()]:
                    continue
                tx.append(wv.word_vec(ent.lower()))

            neighbors_per_triple[tID] = t_neighbors
            tx = np.array(tx)
            # print((csr.csr_matrix(tx)))
        else:
            continue


        print(tID)



        site_links = set()
        for ent in entities_in_triples:
            pages = WikibaseSearchItemPageGenerator(ent, language='en', total=10, site=site)

            for p in pages:
                # print("\n\n\n\nYEAH: ",ent)
                
                

                try:
                    page = pywikibot.Page(site, p.getSitelink(site))
                    if page.isDisambig():
                                page = pywikibot.Page(site, p.getSitelink(site))
                                # print(page.isDisambig())
                                text = page.get()
                                if 'See also' in text:
                                    text = text[:text.index('See also')]
                                interlinks = find_interlinks(text)
                                mentions = [r[1].lower() for r in interlinks.items()]
                                disambig_links = [r[0] for r in interlinks.items()]
                                print(disambig_links)
                                # print(page.get())
                                site_links.update(disambig_links)
                    else:
                        site_links.add(p.getSitelink(site))

                except:
                    pass

        ents_for_training = list()
        adjacency = defaultdict(list)    
        x = list()
        y = list()
        
        print(site_links)
        if len(site_links) not in  list(range(25,40)):
            continue
        for link in site_links:
            ents_per_page = set()
            neighbors = defaultdict(set)
            entity_to_id = {}
            id_to_entity = defaultdict(list)
            try:
                page = pywikibot.Page(site, link)
                text = page.get()
                site_id = page.data_item().getID()
            except:
                continue
            if skip_page(page):
                continue
            aliases = []
            interlinks = find_interlinks(text)
            mentions = [r[1].lower() for r in interlinks.items()]
            reverse_interlinks = {r[1].lower() : r[0] for r in interlinks.items()}
        #     print(reverse_interlinks)
            try:
                aliases = [al.lower() for al in page.data_item().aliases['en']]
            except:
                pass
            aliases.append(link.lower())

            for al in aliases:
                interlinks[al] = link
                mentions.append(al.lower())
                mentions.extend(al.lower().split())
                reverse_interlinks[al.lower()] = link
                entity_to_id[al.lower()]=site_id
                id_to_entity[site_id].append(al.lower())

                for bit in al.lower().split():
                    entity_to_id[bit] = site_id
                    reverse_interlinks[bit] = link
                    id_to_entity[site_id].append(bit)
                    id_to_entity[site_id].append(bit)
                neighbors[al].update(aliases)

            ents_per_page.update(aliases)
            if '== Notes ==' in text:
                text = text[:text.index('== Notes ==')]
            if '== References ==' in text:
                text = text[:text.index('== References ==')]
            clean_text = filter_wiki(text, remove_newline=False)
            paragraphs = clean_text.split('\n')
            while ''   in paragraphs:
                paragraphs.remove('')

            for paragraph in paragraphs:
                tokens = nlp(paragraph)
                new_ents = set()
                for t in tokens:
                    if t.ent_type:
                        if t.ent_type_ in keep_ents:
                            new_ents.add(t.text.lower())

                ents_per_paragraph = set()
                for ent in new_ents:
                    if ent not in wv:
                        continue
                    if ent in mentions:

                        if ent in entity_to_id:
                            ents_per_paragraph.add(ent)
                            # print(ent, ": mention linked already")
                            continue
                        page = pywikibot.Page(site, reverse_interlinks[ent])
                        try:
                            qid = page.data_item().getID()
                        except:
                            continue
                        if ent in entity_to_id:
                            if entity_to_id[ent] != qid:
                                print("Houston we have a problem")
                        # print("exact match:",ent, reverse_interlinks[ent],  page.getID() )

                        entity_to_id[ent] = qid

                        id_to_entity[qid].append(ent)
                        ents_per_paragraph.add(ent)

                    else:
                        sim = max([textdistance.jaccard(i.split(), ent.split()) for i in mentions])
                        if sim >= .85:
                            ind = argmax([textdistance.jaccard(i.split(), ent.split()) for i in mentions])

                            # print("approx match:", ent,mentions[ind],'%%%', reverse_interlinks[mentions[ind]], max([textdistance.jaro_winkler(i, ent) for i in mentions]) )
                            page = pywikibot.Page(site, reverse_interlinks[mentions[ind]])
                            if skip_page(page):
                                continue
                            try:
                                qid = page.data_item().getID()
                            except:
                                continue
                            if ent in entity_to_id:
                                if entity_to_id[ent] != qid:
                                    print("Houston we have a problem")
                            entity_to_id[ent] = qid
                            ents_per_paragraph.add(ent)

                ents_per_paragraph = list(ents_per_paragraph)
                if len(ents_per_paragraph) < 2:
                    continue
                ents_per_page.update(ents_per_paragraph)
                for i in range(len(ents_per_paragraph)):
                    neighbors[ents_per_paragraph[i]].update(ents_per_paragraph)
                
            

            ents_per_page = list(ents_per_page)
            length = len(x)
            x.extend(ents_per_page)
            y.extend([entity_to_id[ents_per_page[i]] for i in range(len(ents_per_page))])
            print("Lenghts " , len(x), len([i[0] for i in neighbors.items()]))
            if len(x) != len([i[0] for i in neighbors.items()]):
                for e in ents_per_page:
                    if e not in neighbors:
                        print(e, neighbors)
            for item in neighbors.items():
                adjacency[ents_per_page.index(item[0])+length] = [ents_per_page.index(i) +length for i in item[1]]

        # print(x,y)
        


        allx, ally, train_x, train_y, tx, ty, adjacency, id_map = create_dataset(x,y, tx,tID, adjacency)
        res = run_nn(str(tID), 'dense')
        test_output = res[:2]
        print("Triple id: ",str(tID), ' triple:', data_item['triple'])
        site = pywikibot.Site( 'en','wikipedia')  # any site will work, this is just an example
        repo = site.data_repository()
        print(id_map[test_output[0]],  id_map[test_output[1]])
        item = pywikibot.ItemPage(repo, id_map[test_output[0]])
        try:
            label_s = item.get()['claims']['P646'][0].target
        except:
            label_s = id_map[test_output[0]]
        item = pywikibot.ItemPage(repo, id_map[test_output[1]])
        try:
            label_o = item.get()['claims']['P646'][0].target
        except:
            label_o = id_map[test_output[1]]
  
        print("True labels: ",data_item['true_link'])
        print('\nOutput: \nSubject: ', label_s, id_to_entity[id_map[test_output[0]]])
        print('\nOutput: \nObject: ', label_o, id_to_entity[id_map[test_output[1]]])
        f= open("results_gcn.txt","a+")

        f.write("Triple id: "+ str(tID)+"   "+" ".join(data_item['triple'])+'\n')
        f.write(" DENSE: Sub "+data_item['true_link']['subject']+"  " +label_s +" Obj "+data_item['true_link']['object'] +"  "+ label_o+" "+id_map[test_output[0]]+"  "+ id_map[test_output[1]]+"\n")
        # f.close()
        if data_item['true_link']['subject'] == label_s:
            correct_subs += 1
        if data_item['true_link']['object'] == label_o:
            correct_obj += 1
        if data_item['true_link']['subject'] == label_s and data_item['true_link']['object'] == label_o:
            correct_both += 1
        tot += 1
        
    # f= open("results.txt","a+")
        f.write("Final results: total triples "+ str(tot)+ "  Correct subs: "+ str(correct_subs)+ "  correct obj: " + str(correct_obj)+ "  both  "+ str(correct_both)+"\n")
        f.close()
        ##### GCN ####
        res = run_nn(str(tID), 'gcn_cheby')
        test_output = res[:2]
        print("Triple id: ",str(tID), ' triple:', data_item['triple'])
        site = pywikibot.Site( 'en','wikipedia')  # any site will work, this is just an example
        repo = site.data_repository()
        print(id_map[test_output[0]],  id_map[test_output[1]])
        item = pywikibot.ItemPage(repo, id_map[test_output[0]])
        try:
            label_s = item.get()['claims']['P646'][0].target
        except:
            label_s = id_map[test_output[0]]
        item = pywikibot.ItemPage(repo, id_map[test_output[1]])
        try:
            label_o = item.get()['claims']['P646'][0].target
        except:
            label_o = id_map[test_output[1]]
  
        print("True labels: ",data_item['true_link'])
        print('\nOutput: \nSubject: ', label_s, id_to_entity[id_map[test_output[0]]])
        print('\nOutput: \nObject: ', label_o, id_to_entity[id_map[test_output[1]]])
        f= open("results_gcn.txt","a+")

        f.write("Triple id: "+ str(tID)+"   "+" ".join(data_item['triple'])+'\n')
        f.write(" GCN: Sub "+data_item['true_link']['subject']+"  " +label_s +" Obj "+data_item['true_link']['object'] +"  "+ label_o+" "+id_map[test_output[0]]+"  "+ id_map[test_output[1]]+"\n")
        # f.close()
        if data_item['true_link']['subject'] == label_s:
            g_correct_subs += 1
        if data_item['true_link']['object'] == label_o:
            g_correct_obj += 1
        if data_item['true_link']['subject'] == label_s and data_item['true_link']['object'] == label_o:
            g_correct_both += 1
        
    # f= open("results.txt","a+")
        f.write("Final results: total triples "+ str(tot)+ "  Correct subs: "+ str(g_correct_subs)+ "  correct obj: " + str(g_correct_obj)+ "  both  "+ str(g_correct_both)+"\n")
        f.close()


def create_dataset(xx, yy, tx, tID, adjacency):
    wv = KeyedVectors.load("fasttext-model.kv", mmap='r')

    all_ids =list(set(yy))

    x = np.ndarray(shape=(len(xx),300))

    ally = np.zeros(shape=(len(yy)+tx.shape[0],len(all_ids)) )
    y = np.zeros(shape=(len(yy),len(all_ids)) )
    for i in range(len(xx)):
        x[i]=wv.word_vec(xx[i])
        ally[i][all_ids.index(yy[i])] = 1
        y[i][all_ids.index(yy[i])] = 1

    # allx = csr.csr_matrix(allx)
    # print(xx.shape, tx.shape)
    allx = np.concatenate((x,tx))
    # print(allx.shape)
    allx = csr.csr_matrix(allx)
    x = csr.csr_matrix(x)

    q = [i for i in range(x.shape[0],allx.shape[0])]
    
    for i in range(x.shape[0],allx.shape[0]):
        if i == x.shape[0]:
            adjacency[i] = q
            
        elif i == x.shape[0]+1:
            adjacency[i] = q
        else:
            adjacency[i] = q[:2]

    tx = csr.csr_matrix(tx)
    ty = np.zeros(shape=(tx.shape[0],len(all_ids)) )
    index =  [i + allx.shape[0] for i in range(tx.shape[0])]
    files = ['allx', 'ally', 'x', 'y', 'tx', 'ty', 'graph', 'index','ids']
    for f in files:
        if f == 'allx':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(allx , filehandle)
        if f == 'ally':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(ally , filehandle)
        if f == 'x':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(x , filehandle)
        if f == 'y':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(y , filehandle)
        if f == 'tx':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(tx , filehandle)
        if f == 'ty':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(ty , filehandle)
        if f == 'graph':
            filename = 'data/ind.'+str(tID)+'.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(adjacency , filehandle)
        if f == 'index':
            filename = 'data/ind.'+str(tID)+'.test.'+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(index , filehandle)
        if f == 'ids':
            filename = 'data/ind.'+str(tID)+"."+f
            with open(filename, 'wb') as filehandle:
                pickle.dump(all_ids , filehandle)
    return allx, ally, x, y, tx, ty, adjacency, all_ids




# def run_nn(allx, ally, x, y, tx, ty, adjacency):
def run_nn(filename, model):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()    
        keys_list = [keys for keys in flags_dict]    
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    del_all_flags(tf.app.flags.FLAGS)
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    # flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    FLAGS = tf.app.flags.FLAGS
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_from_file(filename)
    # load_data(x, y, adjacency, tx = tx, ty = ty, x = x, y = y)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()


    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        
        # pred = sess.run(model.predict(), feed_dict=feed_dict_val)
        # for el in pred:
        #     print(list(el).index(1.))
        return outs_val[0], outs_val[1], (time.time() - t_test)


    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #     print("Early stopping...")
        #     break

    print("Optimization Finished!")
    with open("data/ind.{}.ids".format(filename), 'rb') as f:

        ids = pkl.load(f)
    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    feed_dict_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
    pred = sess.run(model.predict(), feed_dict=feed_dict_val)

    results = np.array([np.argmax(i) for i in pred])
    print(y_test.shape)
    for i in range(len(results[test_mask])):
        print(ids[results[test_mask][i]])
    # print(ids[results[val_mask][-2]], ids[results[val_mask][-1]])
    return results[test_mask]

# query()
run_nn('nyt', 'dense')
run_nn('nyt', 'gcn_cheby')