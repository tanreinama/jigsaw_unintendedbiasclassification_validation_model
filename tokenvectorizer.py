import re
import os
import json
import pickle
import bz2
import numpy as np
from tqdm import tqdm
from fastcache import clru_cache as lru_cache
from textblob import Word
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from multiprocessing import Pool

def trans_text(s, dict):
    return s.translate(dict)
tknzr = TweetTokenizer(strip_handles=True)
reg_twt = re.compile(r"(https:\/\/twitter[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")
reg_wik = re.compile(r"(https:\/\/[a-z][a-z]\.wikipedia\.[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")
reg_utb1 = re.compile(r"(https:\/\/youtu\.be[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")
reg_utb2 = re.compile(r"(https:\/\/www\.youtube\.com[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")
reg_url1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)")
reg_url2 = re.compile(r"(bit\.ly\/[a-zA-Z0-9]+)")
reg_url3 = re.compile(r"(goo\.gl\/[a-zA-Z0-9]+)")
reg_url4 = re.compile(r"(www\.[a-zA-Z][a-zA-Z0-9\.]*[a-zA-Z0-9]+\.(com|org|net|int|edu|gov|mil|([a-z][a-z])))")
reg_tid = re.compile(r"(@[a-zA-Z0-9]+)")
reg_tag = re.compile(r"(#[a-zA-Z0-9]+)")
reg_num = re.compile(r"([0-9][0-9]+)")
prefix_word_list = ['trans','super','hyper','every','cyber','anti','bull','and','any','pre','non','sub','car','dis','mis','neo','at','un','ex','be','co','re','de','by']
emoji_isolate_dict = {'(8':'🙂', '(:':'🙂', '(:>':'🙂', '(;':'🙂', '(=':'🙂', '(=>':'🙂', ')\';':'🙁', ')8':'🙂', '):':'🙁', ');':'🙁', ')=':'🙁', ':\'(':'🙂', ':\')':'🙂', ':\'D':'🙂', ':\'d':'😜', ':(':'🙁', ':)':'🙂', ':*(':'🙁', ':*)':'🙂', ':8':'🙂', ':;':'🙁', ':=':'🙂', ':D':'🙂',':O(':'🙂', ':O)':'🙂', ':Op':'😜', ':P':'😜', ':[':'🙁', ':\'':'🙁', ':]':'🙂', ':d':'😜', ':o':'🙂', ':o(':'🙁', ':o)':'🙂', ':oD':'🙂', ':o\'':'🙁', ':op':'😜', ':p':'😜', ':|':'🙁', ':}':'🙂', ';\')':'🙂', ';(':'🙁', ';)':'🙂', ';:':'🙁', ';D':'🙂', ';O)':'🙂', ';P':'😜', ';[':'🙁', ';]':'🙂', ';d':'😜', ';o(':'🙁', ';o)':'🙂', ';oD':'🙂', ';oP':'😜', ';od':'😜', ';op':'😜', ';o}':'🙂', ';p':'😜', ';{':'🙁', ';}':'🙂', '=(':'🙁', '=)':'🙂', '=:':'🙁', '=D':'🙂', '=Op':'😜', '=P':'😜', '=\'':'🙁', '=d':'😜', '=o)':'🙂', '=op':'🙁', '=o}':'🙂', '=p':'😜', '=}':'🙂', '>:':'🙁', '>:(':'🙁', '>:)':'🙁', '>;':'🙂', '>=p':'😜', '@:':'🙂', '@;':'🙁', '{:':'🙂', '{;':'🙂', '{=':'🙂', '|:':'🙁', '}:':'🙁', '};':'🙁'}
separatedict = {ord(k):' ' for k in "!\"\'-#$%&()*+/:,.;=@[\\]^_`{|}~\t\r\n"}
def replace_text(s, deletedict, isolatedict):
    s = re.sub(reg_twt, ' <twt> ' ,s)
    s = re.sub(reg_wik, ' <wik> ' ,s)
    s = re.sub(reg_utb1, ' <utb> ' ,s)
    s = re.sub(reg_utb2, ' <utb> ' ,s)
    s = re.sub(reg_url1, ' <url> ' ,s)
    s = re.sub(reg_url2, ' <url> ' ,s)
    s = re.sub(reg_url3, ' <url> ' ,s)
    s = re.sub(reg_url4, ' <url> ' ,s)
    s = re.sub(reg_tid, ' <tid> ' ,s)
    s = re.sub(reg_tag, ' <tag> ' ,s)
    s = s.replace('?', ' <qes> ')
    s = s.replace('\n', ' <ret> ')
    s = re.sub(reg_num, ' <num> ' ,s)
    s = s.translate(deletedict)
    return s.translate(isolatedict)

def getsimbols(s):
    all_charactors = set()
    for cc in str(s):
        if ord(cc) >= 256:
            all_charactors.add(cc)
    return all_charactors

stemmer = PorterStemmer()
@lru_cache(120000)
def stem(s):
    return stemmer.stem(s)
@lru_cache(120000)
def corr(s):
    return str(Word(s).correct())
def get_til(s, dict_words):
    if s in dict_words:
        return s
    s_t = s.title()
    if s_t in dict_words:
        return s_t
    s_l = s.lower()
    if s_l in dict_words:
        return s_l
    return None
def get_stil(s, dict_words):
    t = get_til(s, dict_words)
    if t is not None:
        return t
    s = stem(s)
    t = get_til(s, dict_words)
    if t is not None:
        return t
def get_heads(s, pref, dict_words):
    p = pref + '-' + s[len(pref):]
    t = get_stil(p, dict_words)
    if t is not None:
        return [t]
    p = pref + '-' + s[len(pref):].lower()
    t = get_stil(p, dict_words)
    if t is not None:
        return [t]
    p = pref + '-' + s[len(pref):].title()
    t = get_stil(p, dict_words)
    if t is not None:
        return [t]
    p = pref + '-' + stem(s[len(pref):])
    t = get_til(p, dict_words)
    if t is not None:
        return [t]
    p = pref + '-' + stem(s[len(pref):].lower())
    t = get_til(p, dict_words)
    if t is not None:
        return [t]
    p = pref + '-' + stem(s[len(pref):].title())
    t = get_til(p, dict_words)
    if t is not None:
        return [t]
    t1 = get_til(pref, dict_words)
    t2 = get_stil(s[len(pref):], dict_words)
    if t1 is not None and t2 is not None:
        return [t1,t2]
    return None
def get_stem(s, dict_words, spell_collector):
    t = get_stil(s, dict_words)
    if t is not None:
        return [t]
    if '-' in s:
        t = get_stil(s.replace('-',''), dict_words)
        if t is not None:
            return [t]
    if '\'' in s:
        t = get_stil(s.replace('\'',''), dict_words)
        if t is not None:
            return [t]
    for pref in prefix_word_list:
        if len(s) > len(pref)+1 and s.lower().startswith(pref):
            tl = get_heads(s, pref, dict_words)
            if tl is not None:
                return tl
    if type(spell_collector) == dict:
        if s in spell_collector:
            p = spell_collector[s]
            t = get_stil(p, dict_words)
            if t is not None:
                return [t]
    elif str(type(spell_collector)) == "<class 'function'>":
        p = spell_collector(s)
        t = get_stil(p, dict_words)
        if t is not None:
            return [t]
    elif type(spell_collector) == str and spell_collector == 'textblob':
        p = corr(s)
        t = get_stil(p, dict_words)
        if t is not None:
            return [t]
    return None
def fix_spell(s, dict_words, spell_collector):
    if s in emoji_isolate_dict:
        return [emoji_isolate_dict[s]]
    tl = get_stem(s, dict_words, spell_collector)
    if tl is not None:
        return tl
    t = s.translate(separatedict)
    if t != s:
        v = [get_stem(u, dict_words, spell_collector) for u in t.split()]
        return sum([w for w in v if w is not None],[])
    return [s]
def tokenize_text(s, dict_words, spell_collector):
    return sum([fix_spell(w, dict_words, spell_collector) for w in tknzr.tokenize(s)],[])

class TokenVectorizer:

    def __init__(self, vector_dict, uniform_unknown_word=False, lemma_dict={}, spell_collector='textblob', num_process=-1):
        self.vector_dict = vector_dict
        self.uniform_unknown_word = uniform_unknown_word
        self.lemma_dict = lemma_dict
        self.spell_collector = spell_collector
        self.num_process = num_process if num_process>0 else os.cpu_count()
        self.word_index = dict()
        self.unknown_words = set()
        if type(self.vector_dict) == str:
            self.load_vector_dict(self.vector_dict)
        if type(self.lemma_dict) == str:
            if os.path.isfile(self.lemma_dict):
                with open(self.lemma_dict) as f:
                    self.lemma_dict = json.load(f)
            else:
                self.lemma_dict = dict()
        if type(self.spell_collector) == str:
            if os.path.isfile(self.spell_collector):
                with open(self.spell_collector) as f:
                    self.spell_collector = json.load(f)
        self.deletedict = dict()
        self.isolatedict = dict()

    def load_vector_dict(self, vector_file):
        if vector_file.endswith('.pickle'):
            with open(vector_file, 'rb') as f:
                self.vector_dict = pickle.load(f)
        elif vector_file.endswith('.bz2'):
            with bz2.BZ2File(vector_file, 'rb') as f:
                self.vector_dict = pickle.loads(f.read())
        else:
            def get_coefs(word, *arr):
                return word, np.asarray(arr, dtype='float32')
            with open(vector_file) as f:
                self.vector_dict = dict(get_coefs(*line.strip().split(' ')) for line in f)
        if '<url>' not in self.vector_dict:
            v = self.vector_dict.values()
            mat_size = np.max([len(t) for t in v])
            emb_mean, emb_std = np.mean([np.mean(t) for t in v]), np.mean([np.std(t) for t in v])
            np.random.seed(12)
            self.vector_dict['<tid>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<tag>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<qes>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<ret>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<twt>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<wik>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<utb>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<url>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
            self.vector_dict['<num>'] = np.random.normal(emb_mean, emb_std, (mat_size,))
    def preprocess(self, string_list):
        lenmadict = {ord(k):v for k,v in self.lemma_dict.items()}
        all_simbols = set()
        dict_words = set(self.vector_dict.keys())
        if self.num_process <= 1:
            if len(lenmadict) > 0:
                string_list = [s.translate(lenmadict) for s in string_list]
            for s in string_list:
                all_simbols |= getsimbols(s)
        else:
            if len(lenmadict) > 0:
                with Pool(self.num_process) as pl:
                    r = pl.starmap(trans_text, [ (s,lenmadict) for s in string_list ], chunksize=10000)
                    for i,p in enumerate(r):
                        string_list[i] = p
            with Pool(self.num_process) as pl:
                r = pl.imap_unordered(getsimbols, string_list, chunksize=10000)
                for p in r:
                    all_simbols |= p
        self.deletedict = {ord(k):'' for k in (all_simbols - dict_words)}
        self.isolatedict = {ord(k):' %s '%k for k in (dict_words & all_simbols)}
        if self.num_process <= 1:
            for i,s in enumerate(string_list):
                string_list[i] = replace_text(s,self.deletedict,self.isolatedict)
        else:
            with Pool(self.num_process) as pl:
                r = pl.starmap(replace_text, [ (s,self.deletedict,self.isolatedict) for s in string_list ], chunksize=10000)
                for i,p in enumerate(r):
                    string_list[i] = p
        return string_list


    def tokenize(self, string_list, maxlen=-1, pad_sequence=True):
        dict_words = set(self.vector_dict.keys())

        all_seq = []
        for s in string_list:
            all_seq.append([])
        if self.num_process <= 1:
            for i,s in enumerate(string_list):
                all_seq[i] = tokenize_text(s, dict_words, self.spell_collector)
        else:
            with Pool(self.num_process) as pl:
                r = pl.starmap(tokenize_text, [ (s, dict_words, self.spell_collector) for s in string_list ], chunksize=10000)
                for i,t in enumerate(r):
                    all_seq[i] = t

        for i,s_token in enumerate(all_seq):
            tokens = []
            for t in s_token:
                if t in self.word_index:
                    tokens.append(self.word_index[t])
                else:
                    self.word_index[t] = len(self.word_index) + 1
                    tokens.append(self.word_index[t])
            all_seq[i] = tokens

        if pad_sequence:
            if maxlen <= 0:
                maxlen = np.max([len(t) for t in all_seq])
            for i,tokens in enumerate(all_seq):
                if len(tokens) > maxlen:
                    all_seq[i] = tokens[len(tokens)-maxlen:]
                elif len(tokens) < maxlen:
                    all_seq[i] = [0 for w in range(maxlen-len(tokens))] + tokens
            return np.array(all_seq, dtype=np.int64)
        else:
            if maxlen > 0:
                for i,tokens in enumerate(all_seq):
                    if len(tokens) > maxlen:
                        all_seq[i] = tokens[len(tokens)-maxlen:]
            return all_seq

    def vectorize(self, tokens):
        v = self.vector_dict.values()
        mat_size = np.max([len(t) for t in v])
        if self.uniform_unknown_word:
            emb_mean, emb_std = np.mean([np.mean(t) for t in v]), np.mean([np.std(t) for t in v])
        embedding_matrix = np.zeros((len(self.word_index) + 1, mat_size))
        for word_o, i in self.word_index.items():
            if word_o in self.vector_dict:
                embedding_matrix[i] = self.vector_dict[word_o]
            else:
                if self.uniform_unknown_word:
                    embedding_matrix[i] = np.random.normal(self.emb_mean, self.emb_std, (mat_size,))
                self.unknown_words.add(word_o)
        return embedding_matrix

    def __call__(self, string_list, maxlen=-1, pad_sequence=True):
        print("Preprocess.")
        string_list = self.preprocess(string_list)
        print("Tokenize.")
        tokens = self.tokenize(string_list,maxlen,pad_sequence)
        print("Vectorize.")
        return tokens, self.vectorize(tokens)
