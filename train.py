import tensorflow as tf
import numpy as np
import os
import cv2
from random import shuffle

class mnist(object):
    learning_rate = 0.001
    input_node_name = 'input'
    output_node_name = 'output'
    num_classes = 10
    train_set = []
    test_set = []
    def __init__(self, is_training=True):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3], name=self.input_node_name)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes])
        self.get_list()
        self.network()       
        self.train()
        self.summary()
        self.saver = tf.train.Saver()        
        self.init = tf.global_variables_initializer()

    def network(self):
        conv_1 = tf.layers.conv2d(inputs=self.x, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2,2], strides=2)
        conv_2 = tf.layers.conv2d(inputs=pool_1, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2,2], strides=2)
        conv_3 = tf.layers.conv2d(inputs=pool_2, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool_3 = tf.layers.max_pooling2d(inputs=conv_3, pool_size=[2,2], strides=2, padding='same')
        flatten = tf.layers.flatten(pool_3)
        fully = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fully, 10)
        self.outputs = tf.nn.softmax(self.logits, name=self.output_node_name)
    
    def train(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.correct_pred = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.test_accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def summary(self): 
        summary_train_loss = tf.summary.scalar(name="train", tensor=self.loss, family="loss")
        summary_train_accuracy = tf.summary.scalar(name="train", tensor=self.accuracy, family="accuracy")
        summary_test_loss = tf.summary.scalar(name="test", tensor=self.test_loss, family="loss")
        summary_test_accuracy = tf.summary.scalar(name="test", tensor=self.test_accuracy, family="accuracy")
        self.merged_summary_train_op = tf.summary.merge([summary_train_loss, summary_train_accuracy])
        self.merged_summary_test_op = tf.summary.merge([summary_test_loss, summary_test_accuracy])

    def get_list(self):
        for root, dirs, files in os.walk('dataset/trainingSet'):
            for file in files:
                label = root.split('/')[-1]
                dic = {'label':label, 'file':root+"/"+file}
                self.train_set.append(dic)
        for root, dirs, files in os.walk('dataset/validation'):
            for file in files:
                if file == '.DS_Store':continue
                label = root.split('/')[-1]
                dic = {'label':label, 'file':root+"/"+file}
                self.test_set.append(dic)

    def get_train_image(self, batch_size=64):
        batch_features = []
        labels = []
        while True:
            shuffle(self.train_set)
            for data in self.train_set:
                image = cv2.imread(data['file'], cv2.IMREAD_COLOR)
                resize_image = cv2.resize(image, (28,28), interpolation=cv2.INTER_CUBIC)
                b,g,r = cv2.split(resize_image)
                rgb_img = cv2.merge([r,g,b])
                rgb_img = rgb_img/255.0
                batch_features.append(rgb_img)
                label = self.dense_to_one_hot(int(data['label']), self.num_classes)
                labels.append(label)
                if len(batch_features) >= batch_size:
                    yield np.array(batch_features), np.array(labels)
                    batch_features = []
                    labels = []

    def get_test_image(self, batch_size=64):
        batch_features = []
        labels = []
        while True:
            shuffle(self.test_set)
            for data in self.test_set:
                image = cv2.imread(data['file'], cv2.IMREAD_COLOR)
                resize_image = cv2.resize(image, (28,28), interpolation=cv2.INTER_CUBIC)
                b,g,r = cv2.split(resize_image)
                rgb_img = cv2.merge([r,g,b])
                rgb_img = rgb_img/255.0
                batch_features.append(rgb_img)
                label = self.dense_to_one_hot(int(data['label']), self.num_classes)
                labels.append(label)
                if len(batch_features) >= 100:
                    yield np.array(batch_features), np.array(labels)
                    batch_features = []
                    labels = []

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        return np.eye(num_classes)[labels_dense]

    def save_graph_to_file(self, sess, path): 
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [self.output_node_name])
        with tf.gfile.FastGFile(path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

MODEL_NAME = 'mnist'
batch_size = 128
training_iters = 271
display_step = 27
mnist_net = mnist(is_training=True)
with tf.Session() as sess:
    sess.run(mnist_net.init)
    #mnist_net.saver.restore(sess, "output/mnist")
    tf.train.write_graph(sess.graph_def, 'output', MODEL_NAME + '.pbtxt', True)
    summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
    step = 1
    while step < training_iters:
        train_batch = mnist_net.get_train_image(batch_size)
        batch_x, batch_y = next(train_batch)
        test_batch = mnist_net.get_test_image(batch_size)
        batch_a, batch_b = next(test_batch)
        sess.run(mnist_net.optimizer, feed_dict={mnist_net.x: batch_x, mnist_net.y: batch_y})
        if step % display_step == 0:
            train_acc = sess.run(mnist_net.accuracy, feed_dict={mnist_net.x: batch_x, mnist_net.y: batch_y})
            train_loss = sess.run(mnist_net.loss, feed_dict={mnist_net.x: batch_x, mnist_net.y: batch_y})
            test_acc = sess.run(mnist_net.test_accuracy, feed_dict={mnist_net.x: batch_a, mnist_net.y: batch_b})
            test_loss = sess.run(mnist_net.test_loss, feed_dict={mnist_net.x: batch_a, mnist_net.y: batch_b})
            print("Iter " + str(step) + ", Training Loss = " + \
                 "{:.6f}".format(train_loss) + ", Training Accuracy = " + \
                 "{:.5f}".format(train_acc) + ", Testbatch Loss = " + \
                 "{:.6f}".format(test_loss) + ", Testing Accuracy = " + \
                 "{:.5f}".format(test_acc))
        step = step + 1
        summary = sess.run(mnist_net.merged_summary_train_op, feed_dict={mnist_net.x: batch_x, mnist_net.y: batch_y})
        summary_writer.add_summary(summary, step)
        summary = sess.run(mnist_net.merged_summary_test_op, feed_dict={mnist_net.x: batch_a, mnist_net.y: batch_b})
        summary_writer.add_summary(summary, step)
    mnist_net.saver.save(sess, 'output/' + MODEL_NAME)
    mnist_net.save_graph_to_file(sess, 'output/'+MODEL_NAME+'.pb')
    print("Finish")
