#! -*- coding:utf-8 -*-
# tensorflow 1.2 + Keras 2.0.6

import numpy as np
import pandas as pd
import jieba

pos = pd.read_excel('pos.xls', header=None)
neg = pd.read_excel('neg.xls', header=None)
pos['words'] = pos[0].apply(jieba.lcut)
neg['words'] = neg[0].apply(jieba.lcut)

words = {}
for l in pos['words'].append(neg['words']):
  for w in l:
    if w in words:
      words[w] += 1
    else:
      words[w] = 1

min_count = 10
maxlen = 100

words = {i: j for i, j in words.items() if j >= min_count}
id2word = {i + 1: j for i, j in enumerate(words)}
word2id = {j: i for i, j in id2word.items()}


def doc2num(s):
  s = [word2id.get(i, 0) for i in s[:maxlen]]
  return s + [0] * (maxlen - len(s))


pos['id'] = pos['words'].apply(doc2num)
neg['id'] = neg['words'].apply(doc2num)

x = np.vstack([np.array(list(pos['id'])), np.array(list(neg['id']))])
y = np.array([[1]] * len(pos) + [[0]] * len(neg))


idx = range(len(x))
np.random.shuffle(idx)
x = x[idx]
y = y[idx]

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Lambda
from keras.layers import LSTM
from keras import backend as K


input = Input(shape=(None,))
input_vecs = Embedding(len(words) + 1, 128, mask_zero=True)(input)
lstm = LSTM(128, return_sequences=True, return_state=True)(input_vecs)
lstm_state = Lambda(lambda x: x[1])(lstm)
dropout = Dropout(0.5)(lstm_state)
predict = Dense(1, activation='sigmoid')(dropout)


lstm_sequence = Lambda(lambda x: K.concatenate(
    [K.zeros_like(x[0])[:, :1], x[0]], 1))(lstm)
lstm_dist = Lambda(lambda x: K.sqrt(K.sum(
    (x[0] - K.expand_dims(x[1], 1))**2, 2) / K.sum(x[1]**2, 1, keepdims=True)))([lstm_sequence, lstm_state])

model = Model(inputs=input, outputs=predict)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_dist = Model(inputs=input, outputs=lstm_dist)
model_dist.compile(loss='mse',
                   optimizer='adam')

batch_size = 128
train_num = 15000

model.fit(x[:train_num], y[:train_num], batch_size=batch_size,
          epochs=5, validation_data=(x[train_num:], y[train_num:]))

import uniout


def saliency(s):
  ws = jieba.lcut(s)[:maxlen]
  x_ = np.array([[word2id.get(w, 0) for w in ws]])
  score = np.diff(model_dist.predict(x_)[0])
  idxs = score.argsort()
  return [(i, ws[i], -score[i]) for i in idxs]  # 输出结果为：(词位置、词、词权重)
