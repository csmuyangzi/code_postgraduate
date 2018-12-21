import tensorflow as tf
import numpy as
from tensorflow.contrib import rnn

#下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./trainingandtestdata/mnist/', one_hot=True)

#每张图片尺寸28*28
#RNN将数据分块，而FNN则是一次性的把数据输入到网络

chunk_size = 28
chunk_n =28

rnn_size = 256

n_output_layer = 10

X = tf.placeholder('float', [None, chunk_n, chunk_size])
#占位符placeholder，我们在运行计算时输入这个值。我们希望能够输入任意数量的图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图
Y = tf.placeholder('float')

#定义神经网络
def recurrent_neural_network(data):
    layer = {'W_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    lstm_cell = rnn.rnn_cell_BasicLSTMCell(rnn_size)

    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(data, chunk_n, 0)
    outputs, status = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])

    return output

#每次使用100条数据训练
batch_size = 100

#训练
def train_neural_network(X, Y):
    predict = recurrent_neural_netword(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.03)minimize(cost_func)

    epochs = 13
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        epoch_loss = 0
        for epoch in range(epochs)
            for i in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size, chunk_n, chunk_size])
                _, c = session.run([optimizer, cost_func], feed_dict = {X:x, Y:y})
                epoch_loss = c
            print(epoch, ':',epoch_loss)

        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率：', accuracy.eval({X:mnist.test.images.reshaoe(-1, chunk_n, chunk_size), Y:mnist.test.labels}))

train_neural_network(X, Y)

