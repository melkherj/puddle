from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import ops
import hashlib

#@author melkherj

def weight_variable(shape,seed=None):
    initial = tf.truncated_normal(shape, stddev=0.1,seed=seed)
    return tf.Variable(initial)
        
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
        
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

def file_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()        

def graph_hash(graphdef):
    filename = "/tmp/temp_graph.pb"
    tf.train.write_graph(graphdef, ".logs", filename, True)
    return file_md5(filename)

def session_hash(sess):
    filename = '/tmp/temp_sess_model.ckpt'
    saver = tf.train.Saver() 
    saver.save(sess,filename)
    return file_md5(filename)

class MNISTModeler:

    def __init__(self,train_n=1000,test_n=1000,seed=None):
        #set random seed
        self.seed = seed

        # load data
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.train_X = self.mnist.train.images[:train_n]
        self.train_Y = self.mnist.train.labels[:train_n]
        self.test_X = self.mnist.test.images[:test_n]
        self.test_Y = self.mnist.test.labels[:test_n]
        
        # define the graph
        self.initialize_graph()
        self.start_session()
       
        # do initial scoring
        self.score_train()

    def graph_hash(self):
        return graph_hash(self.get_graphdef())
    
    def session_hash(self):
        return session_hash(self.sess)

    def initialize_graph(self):
        tf.reset_default_graph()
        self.define_network()
        self.define_eval()
        self.graph = tf.get_default_graph()
        self.set_graph_vars()
        tf.set_random_seed(self.seed)

    def define_network(self):
        # inputs
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        W_conv1 = weight_variable([5, 5, 1, 32],seed=self.seed)
        b_conv1 = bias_variable([32])
        
        x_image = tf.reshape(self.x, [-1,28,28,1])
        
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        
        W_conv2 = weight_variable([5, 5, 32, 64],seed=self.seed)
        b_conv2 = bias_variable([64])
        
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        
        W_fc1 = weight_variable([7 * 7 * 64, 1024],seed=self.seed)
        b_fc1 = bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob,seed=self.seed)

        W_fc2 = weight_variable([1024, 10],seed=self.seed)
        b_fc2 = bias_variable([10])
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='y_conv')

    def define_eval(self):
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv),name='cross_entropy')
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1),
                name='correct_prediction')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32),name='accuracy')
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy,name='train_step')

    def set_graph_vars(self):
        ''' update instance variable helpers that refer to commonly used graph variables '''
        self.y_conv = self.graph.get_tensor_by_name('y_conv:0')
        self.cross_entropy = self.graph.get_tensor_by_name('cross_entropy:0')
        self.correct_prediction = self.graph.get_tensor_by_name('correct_prediction:0')
        self.accuracy = self.graph.get_tensor_by_name('accuracy:0')
        self.train_step = self.graph.get_operation_by_name('train_step')

    def set_graphdef(self,graphdef):
        ''' given graphdef, set current graph and session '''
        tf.import_graph_def(g.as_graph_def())
        self.graph = tf.get_default_graph()

    def start_session(self):
        self.sess = tf.Session(graph=self.graph)
        try:
            self.sess.run(tf.assert_variables_initialized())
        except tf.errors.FailedPreconditionError:
            # some variables not initialized, do that 
            self.sess.run(tf.initialize_all_variables())

    def get_graphdef(self):
        return self.graph.as_graph_def()

    def update_model(self,x,y):
        self.sess.run(self.train_step,
            feed_dict={self.x: x, self.y_: y, self.keep_prob: 0.5})
        self.score_train() #model changed: keep scores up-to-date

    def score_train(self):
        ''' Score the full training set using the current model '''
        self.P = self.sess.run(self.y_conv,
            feed_dict={self.x: self.train_X, self.y_: self.train_Y,  self.keep_prob: 1.0})

    def test_accuracy(self):
        ''' accuracy of current model on the test set '''
        return self.sess.run(self.accuracy,
            feed_dict={self.x: self.test_X, self.y_: self.test_Y, self.keep_prob: 1.0})

