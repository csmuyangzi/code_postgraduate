#! -*- coding:utf-8 -*-
# Keras 2.0 + Tensorflow 1.0
# backoff2005评测集

import numpy as np
import pandas as pd
import pickle
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

#词频词表
class Random_Choice:
    def __init__(self, elements, weights):
        d = pd.DataFrame(zip(elements, weights))
        self.elements, self.weights = [], []
        for i,j in d.groupby(1):
            self.weights.append(len(j)*i)
            self.elements.append(tuple(j[0]))
        self.weights = np.cumsum(self.weights).astype(np.float64)/sum(self.weights)
    def choice(self):
        r = np.random.random()
        w = self.elements[np.where(self.weights >= r)[0][0]]
        return w[np.random.randint(0, len(w))]

#统计字表
words = pd.read_csv('dict.txt', delimiter='\t', header=None, encoding='utf-8')
words[0] = words[0].apply(unicode)
words = words.set_index(0)[1]

try:
    char2id = pickle.load(open('char2id.dic'))
except:
    from collections import defaultdict
    print u'fail to load old char2id.'
    char2id = pd.Series(list(''.join(words.index))).value_counts()
    char2id[:] = range(1, len(char2id)+1)
    char2id = defaultdict(int, char2id.to_dict())
    pickle.dump(char2id, open('char2id.dic', 'w'))

word_size = 128
maxlen = 48
batch_size = 1024

def word2tag(s):
    if len(s) == 1:
        return 's'
    elif len(s) >= 2:
        return 'b'+'m'*(len(s)-2)+'e'

tag2id = {'s':[1,0,0,0,0], 'b':[0,1,0,0,0], 'm':[0,0,1,0,0], 'e':[0,0,0,1,0]}

def data_generator():
    wc = Random_Choice(words.index, words)
    x, y = [], []
    while True:
        n = np.random.randint(1, 17)
        seq = [wc.choice() for i in range(n)]
        tag = ''.join([word2tag(i) for i in seq])
        seq = [char2id[i] for i in ''.join(seq)]
        if len(seq) > maxlen:
            continue
        else:
            seq = seq + [0]*(maxlen-len(seq))
            tag = [tag2id[i] for i in tag]
            tag = tag + [[0,0,0,0,1]]*(maxlen-len(tag))
            x.append(seq)
            y.append(tag)
        if len(x) == batch_size:
            yield np.array(x), np.array(y)
            x, y = [], []

#训练模型
sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(char2id)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True))(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(inputs=sequence, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

try:
    model.load_weights('model.weights')
except:
    print u'fail to load old weights.'

for i in range(100):
    print i
    model.fit_generator(data_generator(), steps_per_epoch=100, epochs=10)
    model.save_weights('model.weights')

#准确率太低，使用viterbi动态规划提高
zy = {'be':0.5,
      'bm':0.5,
      'eb':0.5,
      'es':0.5,
      'me':0.5,
      'mm':0.5,
      'sb':0.5,
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]

def simple_cut(s):
    if s:
        s = s[:maxlen]
        r = model.predict(np.array([[char2id[i] for i in s]+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []

import re
not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!“”]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result