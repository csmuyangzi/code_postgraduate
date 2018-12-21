import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()

n_input_layer = len(lex)

n_layer_1 = 1000
n_layer_2 = 1000
n_output_layer = 3

def neural_network(data):
    #定义第一层权重和偏置
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer,n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    #定义第二层权重和偏置
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1,n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    #定义输出层的权重和偏置
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2,n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    #w*x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)#激活函数
    layer_2 = tf.add(tf.matmul(layer_1,layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2,layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

X = tf.placeholder('float')
def prediction(tweet_text):
    predict = neural_network(X)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(session, 'model.ckpt')#读取模型
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(tweet_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1

        #print (predict.eval(feed_dict={X:[features]})[[val1,val2,val3]])
        res = session.run(tf.argmax(predict.eval(feed_dict={X:[features]}),1))

prediction("I am so mad")