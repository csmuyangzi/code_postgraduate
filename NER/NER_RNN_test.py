#tensorflow 1.2
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

filename = 'test.conll'
vocabulary_size = 0

def build_dataset(filename):
    global vocabulary_size
    dataset = list()
    labels = list()
    f = open(filename, 'r')
    sentence = f.read().split('\n')
    vocabulary_size = len(sentence)
    for line in sentence:
        word, label = line.split('_')
        if label == 'O':
            label = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        elif label == 'PER':
            label = np.array([0.0, 0.0, 1.0, 0.0], np.float32)
        elif label == 'LOCATION':
            label = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
        elif label == 'ORG':
            label = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        dataset.append(word)
        labels.append(label)
        print word, label
    return dataset, labels

dataset, labels = build_dataset(filename)

print vocabulary_size

def generate_batch(batch_size):
    pass

batch_size = 128
embedding_size = 128  
num_nodes = 4  # (PER, ORG, LOCATION, O)
graph = tf.Graph()
with graph.as_default():
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([embedding_size]))

    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        state = forget_gate * state + input_gate * tf.tanh(update)
        return output_gate * tf.tanh(state), state

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    window_size = 5
    train_inputs = list()
    train_labels = list()
    outputs = list()
    output = saved_output
    state = saved_state

    for _ in xrange(window_size):
        train_data = tf.placeholder(tf.int32, shape=[batch_size])
        embed = tf.nn.embedding_lookup(embeddings, train_data)
        label = tf.placeholder(tf.float32, shape=[num_nodes])
        train_inputs.append(embed)
        train_labels.append(label)
        output, state = lstm_cell(embed, output, state)
    logits = tf.nn.xw_plus_b(outputs, w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels))


   
