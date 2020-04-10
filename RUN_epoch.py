import sys
import codecs
import time
import imp
import os
import tensorflow as tf
import numpy as np
import gensim.models as g
from util import *
from sonnet_model import SonnetModel
import config as cf



def run_epoch(sess, word_batches, model, pname, is_training):

    start_time  = time.time()

    #lm variables
    lm_costs    = 0.0
    total_words = 0
    zero_state  = sess.run(model.lm_initial_state)
    model_state = None
    prev_doc    = -1
    lm_train_op = model.lm_train_op if is_training else tf.no_op()

    word_batch_id  = 0
    train_lm = True


    for bi in range(len(word_batches)):

        if  train_lm:
    
            b = word_batches[word_batch_id]

            #reset model state if it's a different set of documents
            if prev_doc != b[2][0]: 
                model_state = zero_state
                prev_doc = b[2][0]

            feed_dict = {model.lm_x: b[0], model.lm_y: b[1], model.lm_xlen: b[3], model.lm_initial_state: model_state, model.lm_hist: b[5], model.lm_hlen: b[6]}

            cost, model_state, attns, _ = sess.run([model.lm_cost, model.lm_final_state, model.lm_attentions, lm_train_op],
                feed_dict)

            lm_costs    += cost * cf.batch_size #keep track of full cost
            total_words += sum(b[3])

            word_batch_id += 1
        
        

        if (((bi % 10) == 0) and cf.verbose) or (bi == len(word_batches)-1):
            print("\n partition : ", pname)
            print(" Cost : ", np.exp(lm_costs/max(total_words, 1)))

        
            if not is_training and (bi == len(word_batches)-1):

                sys.stdout.flush()

    #return avg batch loss for lm
    return lm_costs/max(word_batch_id, 1)


