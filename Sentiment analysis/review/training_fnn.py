# -*- coding:utf-8 -*-
# python3

import os
import random
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


f = open('lexcion.pickle', 'rb')  # 词汇表
lex = pickle.load(f)
f.close()

'''
#速度太慢
lines = []
lines_sum = []
lines_len = 0
with open('training.csv', encoding = 'latin-1') as f:
    for line in f:
        lines_len += 1
        lines_sum.append(line)
    for m in range(150):
        random_line = random.randint(0,lines_len)
        for i, n in enumerate(lines_sum):
            if i == random_line:
                lines.append(n)
print(len(lines))
'''


def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    file_lines = file.readlines()  # readline不适合大文件
    lines_len = len(file_lines)
    for i in range(n):
        random_line = random.randint(0, lines_len - 1)
        lines.append(file_lines[random_line])
    file.close()
    return lines


'''
#造成list index out of range
def get_random_line(file, point):
    file.seek(point)
    file.readline()#并不能输出完整的一行
    return  file.readline()

#从文件中随机选择n条记录
def get_n_random_line(file_name, n = 150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size
    for i in range(n):
        random_point = random.randint(0, total_bytes-1)#python list区间是左闭右开的，造成list index out of range
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines
'''

"""
# 把字符串转为向量
def string_to_vector(input_file, output_file, lex):
    output_f = open(output_file, 'w')
    lemmatizer = WordNetLemmatizer()
    with open(input_file, buffering=10000, encoding='latin-1') as f:
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]

            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            features = list(features)
            output_f.write(str(label) + ":" + str(features) + '\n')
    output_f.close()


f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()

# lexcion词汇表大小112k,training.vec大约112k*1600000  170G  太大，只能边转边训练了
# string_to_vector('training.csv', 'training.vec', lex)
# string_to_vector('tesing.csv', 'tesing.vec', lex)
"""


def get_test_dataset(test_file):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        #f = f.readlines()
        for line in f:
            label = line.split(':%:%:%')[0]
            tweet = line.split(':%:%:%')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            # 向量化
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y


test_x, test_y = get_test_dataset('testing.csv')

# 定义每个层神经元
n_input_layer = len(lex)
n_layer_1 = 100
n_layer_2 = 100

n_output_layer = 3  # 输出层[0,0,1] [0,1,0] [1,0,0]

# 定义待训练的神经网络


def neural_network(data):
    # 定义第一层权重和偏置
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal(
        [n_input_layer, n_layer_1])), 'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层权重和偏置
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal(
        [n_layer_1, n_layer_2])), 'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层的权重和偏置
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal(
        [n_layer_2, n_output_layer])), 'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w*x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(
        tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float')
Y = tf.placeholder('float')

# 使用数据训练神经网络


def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdadeltaOptimizer(0.03).minimize(
        cost_func)  # 优化，learning rate 默认设为0.001

    # epochs = 13#迭代次数
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver()  # 用来管理所有变量
        i = 0  # 迭代次数
        pre_accuracy = 0
        while True:
            batch_x = []
            batch_y = []
            # if model.ckpt文件已存在:
            # saver.restore(session, 'model.ckpt')  恢复保存的session

            try:
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%')[0]
                    tweet = line.split(':%:%:%')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]
                    # 向量化
                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1

                    batch_x.append(list(features))
                    batch_y.append(eval(label))
                    print(batch_x)
                session.run([optimizer, cost_func], feed_dict={
                            X: batch_x, Y: batch_y})
            except Exception as e:
                print(e)

            # 准确率
            if i % 10 == 0:
                correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy = accuracy.eval({X: test_x, Y: test_y})
                if accuracy > pre_accuracy:  # 保存准确率最高的训练模型
                    print('准确率：', accuracy)
                    pre_accuracy = accuracy
                    saver.save(session, 'model/fnn_model.ckpt')
                i = 0
            i += 1


train_neural_network(X, Y)
