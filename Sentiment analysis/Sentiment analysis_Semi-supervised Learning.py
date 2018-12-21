#! -*- coding:utf-8 -*-
#tensorflow 1.2 + Keras 2.0.6
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.datasets import imdb
from keras import backend as K
import numpy as np

max_features = 10000 #保留前max_features个词
maxlen = 100 #填充/阶段到100词
batch_size = 1000
nb_grams = 10 
nb_train = 1000 #训练样本数

#加载IMDB数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_lm_ = np.append(x_train, x_test)

#构造用来训练语言模型的数据

x_lm = []
y_lm = []
for x in x_lm_:
		for i in range(len(x)):
			x_lm.append([0]*(nb_grams - i + max(0,i-nb_grams))+x[max(0,i-nb_grams):i])
			y_lm.append([x[i]])

x_lm = np.array(x_lm)
y_lm = np.array(y_lm)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x = np.vstack([x_train, x_test])
y = np.hstack([y_train, y_test])

#合并原来的训练集和测试集，随机挑选1000个样本，作为新的训练集，剩下为测试集
idx = range(len(x))
np.random.shuffle(idx)
x_train = x[idx[:nb_train]]
y_train = y[idx[:nb_train]]
x_test = x[idx[nb_train:]]
y_test = y[idx[nb_train:]]

embedded_size = 100 #
hidden_size = 1000 

#encoder
inputs = Input(shape=(None,), dtype='int32')
embedded = Embedding(max_features, embedded_size)(inputs)
lstm = LSTM(hidden_size)(embedded)
encoder = Model(inputs=inputs, outputs=lstm)

#完全用ngram模型训练encode部分
input_grams = Input(shape=(nb_grams,), dtype='int32')
encoded_grams = encoder(input_grams)
softmax = Dense(max_features, activation='softmax')(encoded_grams)
lm = Model(inputs=input_grams, outputs=softmax)
#用sparse交叉熵，可以不用事先将类别转换为one hot形式。
lm.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
lm.fit(x_lm, y_lm,
       batch_size=batch_size,
       epochs=5)

#情感分析部分
#固定encoder，后面接一个简单的Dense层
#这时候训练的只有hidden_size+1=1001个参数
#因此理论上来说，少量标注样本就可以训练充分
for layer in encoder.layers:
    layer.trainable=False

sentence = Input(shape=(maxlen,), dtype='int32')
encoded_sentence = encoder(sentence)
sigmoid = Dense(10, activation='relu')(encoded_sentence)
sigmoid = Dropout(0.5)(sigmoid)
sigmoid = Dense(1, activation='sigmoid')(sigmoid)
model = Model(inputs=sentence, outputs=sigmoid)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=250)
model.evaluate(x_test, y_test, verbose=True, batch_size=batch_size)

#数据集扩展，重新训练模型
y_pred = model.predict(x_test, verbose=True, batch_size=batch_size)
y_pred = (y_pred.reshape(-1) > 0.8).astype(int)
xt = np.vstack([x_train, x_test])
yt = np.hstack([y_train, y_pred])

model.fit(xt, yt,
          batch_size=batch_size,
          epochs=10)

#评估模型的效果
model.evaluate(x_test, y_test, verbose=True, batch_size=batch_size)
