import codecs
import operator
import numpy as np
import random
import math
import codecs
import sys
from collections import defaultdict


def load_vocab(corpus, word_minfreq, dummy_symbols):
    idxword, idxchar = [], []
    wordxid, charxid = defaultdict(int), defaultdict(int)
    word_freq, char_freq = defaultdict(int), defaultdict(int)
    wordxchar = defaultdict(list)

    def update_dic(symbol, idxvocab, vocabxid):
        if symbol not in vocabxid:
            idxvocab.append(symbol)
            vocabxid[symbol] = len(idxvocab) - 1 

    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            word_freq[word] += 1
        for char in line.strip():
            char_freq[char] += 1

    #add in dummy symbols into dictionaries
    for s in dummy_symbols:
        update_dic(s, idxword, wordxid)
        update_dic(s, idxchar, charxid)

    #remove low fequency words/chars
    def collect_vocab(vocab_freq, idxvocab, vocabxid):
        for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
            if f < word_minfreq:
                break
            else:
                update_dic(w, idxvocab, vocabxid)

    collect_vocab(word_freq, idxword, wordxid)
    collect_vocab(char_freq, idxchar, charxid)

    #word id to [char ids]
    dummy_symbols_set = set(dummy_symbols)
    for wi, w in enumerate(idxword):
        if w in dummy_symbols:
            wordxchar[wi] = [wi]
        else:
            for c in w:
                wordxchar[wi].append(charxid[c] if c in charxid else charxid[dummy_symbols[2]])

    return idxword, wordxid, idxchar, charxid, wordxchar


def only_symbol(word):
    for c in word:
        if c.isalpha():
            return False

    return True


def remove_punct(string):
    return " ".join("".join([ item for item in string if (item.isalpha() or item == " ") ]).split())


def load_data(corpus, wordxid, idxword, charxid, idxchar, dummy_symbols):

    pad_symbol, end_symbol, unk_symbol = dummy_symbols
    
    nwords     = [] #number of words for each line
    nchars     = [] #number of chars for each line
    word_data  = [] #data[doc_id][0][line_id] = list of word ids; data[doc_id][1][line_id] = list of [char_ids]
    char_data  = [] #data[line_id] = list of char ids
    rhyme_data = [] #list of ( target_word, [candidate_words], target_word_line_id ); word is a list of characters

    def word_to_char(word):
        if word in set([pad_symbol, end_symbol, unk_symbol]):
            return [ wordxid[word] ]
        else:
            return [ charxid[item] if item in charxid else charxid[unk_symbol] for item in word ]


    for doc in codecs.open(corpus, "r", "utf-8"):

        word_lines, char_lines = [[], []], []
        last_words = []

         #reverse the order of lines and words as we are generating from end to start
        for line in reversed(doc.strip().split(end_symbol)):

            if len(line.strip()) > 0:

                word_seq = [ wordxid[item] if item in wordxid else wordxid[unk_symbol] \
                    for item in reversed(line.strip().split()) ] + [wordxid[end_symbol]]

                char_seq = [ word_to_char(item) for item in reversed(line.strip().split()) ] + [word_to_char(end_symbol)]

                word_lines[0].append(word_seq)
                word_lines[1].append(char_seq)
                char_lines.append([ charxid[item] if item in charxid else charxid[unk_symbol] \
                    for item in remove_punct(line.strip())])
                nwords.append(len(word_lines[0][-1]))
                nchars.append(len(char_lines[-1]))

                last_words.append(line.strip().split()[-1])

        if len(word_lines[0]) == 14: #14 lines for sonnets

            word_data.append(word_lines)
            char_data.extend(char_lines)

            last_words = last_words[2:] #remove couplets (since they don't always rhyme)
            for wi, w in enumerate(last_words):
                rhyme_data.append( (word_to_char(w), [ word_to_char(item)
                    for item_id, item in enumerate(last_words[int((wi/4)*4):int((wi/4+1)*4)]) if item_id != (wi%4) ], (11-wi)) )

    return word_data, char_data, nwords, nchars, rhyme_data   #a rajouter dans le return

def print_stats(partition, word_data, nwords, nchars, rhyme_data):
    print(partition, "statistics:")
    print("  Number of documents         =", len(word_data))
    print("  Number of rhyme examples    =", len(rhyme_data))
    print("  Total number of word tokens =", sum(nwords))
    print("  Mean/min/max words per line = %.2f/%d/%d" % (np.mean(nwords), min(nwords), max(nwords)))
    print("  Total number of char tokens =", sum(nchars))
    print( "  Mean/min/max chars per line = %.2f/%d/%d" % (np.mean(nchars), min(nchars), max(nchars)))


def init_embedding(model, idxword):
    word_emb = []
    for vi, v in enumerate(idxword):
        if v in model:
            word_emb.append(model[v])
        else:
            word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
    return np.array(word_emb)


def pad(lst, max_len, pad_symbol):
    if len(lst) > max_len:
        print("\nERROR: padding")
        print("length of list greater than maxlen; list =", lst, "; maxlen =", max_len)
        raise SystemExit
    return lst + [pad_symbol] * (max_len - len(lst))


def get_vowels():
    return set(["a", "e", "i", "o", "u"])


def coverage_mask(char_ids, idxchar):
    vowels = get_vowels()
    return [ float(idxchar[c] in vowels) for c in char_ids ]


def flatten_list(l):
    return [item for sublist in l for item in sublist]





def create_word_batch(data, batch_size, lines_per_doc, nlines_per_batch, pad_symbol, end_symbol, unk_symbol, shuffle_data):

    docs_per_batch = int(len(data) / batch_size) #nombre de batch qu'on a
    batches = [] #On crée une liste
    doc_ids = list(range(len(data))) #liste croissante de 0 à 2684
    
    if shuffle_data:
        random.shuffle(doc_ids)


    for i in range(docs_per_batch):  #environ pour i = 1,...,70 ; on aura besoin de 70 batchs
        for j in range(int(lines_per_doc / nlines_per_batch)): # Quand on est dans un batch, j= 1, ..., 14/2

            docs       = []
            doc_lens   = []
            doc_lines  = []
            x          = []
            y          = []
            xchar      = []
            xchar_lens = []
            hist       = []
            hist_lens  = []

            for k in range(batch_size): #k=1, ..., 32

                d       = doc_ids[i*batch_size+k]  #pour parcourir l'ensemble des documents  #ok
                wordseq = flatten_list(data[d][0][j*nlines_per_batch:(j+1)*nlines_per_batch]) #Ok
                histseq = flatten_list(data[d][0][:j*nlines_per_batch]) #ok

                x.append([end_symbol] + wordseq[:-1])
                y.append(wordseq)

                docs.append(d)
                doc_lens.append(len(wordseq))
                doc_lines.append(range(j*nlines_per_batch, (j+1)*nlines_per_batch))

                hist.append(histseq if len(histseq) > 0 else [unk_symbol])
                hist_lens.append(len(histseq) if len(histseq) > 0 else 1)

            #pad the data
            word_pad_len = max(doc_lens)
            hist_pad_len = max(hist_lens)
            
            for k in range(batch_size):

                x[k] = pad(x[k], word_pad_len, pad_symbol)
                y[k] = pad(y[k], word_pad_len, pad_symbol)
                hist[k] = pad(hist[k], hist_pad_len, pad_symbol)

            batches.append((x, y, docs, doc_lens, doc_lines, hist, hist_lens))


    return batches



def print_lm_attention(bi, b, attentions, idxword, cf):

    print("\n", "="*100)
    for ex in range(cf.batch_size)[-1:]:
        xword = [ idxword[item] for item in b[1][ex] ]
        hword = [ idxword[item] for item in b[7][ex] ]
        print("\nBatch ID =", bi)
        print("Example =", ex)
        print("x_word =", " ".join(xword))
        print("hist_word=", " ".join(hword))
        for xi, x in enumerate(xword):
            print("\nWord =", x)
            print("\tSum dist =", sum(attentions[ex][xi]))
            attn_dist_sort = np.argsort(-attentions[ex][xi])
            print("\t",)
            for hi in attn_dist_sort[:5]:
                print("[%d]%s:%.3f  " % (hi, hword[hi], attentions[ex][xi][hi]))



    def last_syllable(word):

        i = len(word)
        for c in reversed(word):
            i -= 1
            if c in get_vowels():
                break

        return word[i:]




def postprocess_sentence(line):
    cleaned_sent = ""
    for w in line.strip().split():
        spacing = " "
        if w.startswith("'") or only_symbol(w):
            spacing = ""
        cleaned_sent += spacing + w
    return cleaned_sent.strip()
