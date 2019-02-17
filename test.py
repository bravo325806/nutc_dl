import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.python.platform import gfile

np.set_printoptions(precision=3, suppress=True)

def load_graph():
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open('output/mnist.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    return graph

def load_image():
    image = cv2.imread('dataset/testSet/img_1.jpg')
    resize_image = cv2.resize(image, (28,28), interpolation=cv2.INTER_CUBIC)
    b,g,r = cv2.split(resize_image)
    rgb_img = cv2.merge([r,g,b])
    rgb_img = rgb_img/255.0
    return np.array([rgb_img])

if __name__ == '__main__':
    graph = load_graph()
    image = load_image()
    with tf.Session(graph=graph) as sess:
        x = sess.graph.get_tensor_by_name('input:0')
        output = sess.graph.get_tensor_by_name('output:0')
        result = sess.run(output, feed_dict={x:image})
        print(result)
        
