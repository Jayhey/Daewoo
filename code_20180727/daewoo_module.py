import pandas as pd
import os
import tensorflow as tf
import sklearn.metrics as skm
import time
from collections import Counter
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


NUM_CLASSES = 2

# input 설정. 경로 string을 텐서로 바꿈
def set_input(img, label):
	np.random.seed(1234)
	idx = np.random.permutation(len(img))
	tr_idx = idx[:round(0.8 * len(idx))]
	ts_idx = idx[round(0.8 * len(idx)):]

	train_img = img[tr_idx]
	train_label = label[tr_idx]
	test_img = img[ts_idx]
	test_label = label[ts_idx]

	train_img_tensor = tf.constant(train_img)
	train_label_tensor = tf.constant(train_label)
	test_img_tensor = tf.constant(test_img)
	test_label_tensor = tf.constant(test_label)

	return train_img_tensor, train_label_tensor, test_img_tensor, test_label_tensor, tr_idx, ts_idx


# string 텐서를 img 텐서로 변환 후 crop
def input_tensor(img_path, label, crop, RGB, resize):
	img_file = tf.read_file(img_path)
	img_decoded = tf.image.decode_png(img_file)

	if RGB == False:
	    img_decoded = tf.image.rgb_to_grayscale(img_decoded)
	
	img_crop = tf.image.crop_to_bounding_box(img_decoded, 135, 0, 135, 480)
	img_float = tf.to_float(img_crop)
	
	
	if crop == True :
		img_crop = tf.random_crop(img_float, size=[135, 135, 3])
    
		for i in range(1,10):
			_label = tf.one_hot(label, NUM_CLASSES)
			
			_img_crop = tf.image.crop_to_bounding_box(img_decoded, 135, 38*i, 135, 135)
			_img_float = tf.to_float(_img_crop)
			_img_crop = tf.random_crop(_img_float, size=[135, 135, 3])
			
			label_crop = tf.concat([label_crop, _label], axis=0)
			img_crop = tf.concat([img_crop, _img_crop], axis=0)

		return tf.reshape(img_crop, [-1,135,135,3]), tf.reshape(label_crop, [-1,NUM_CLASSES])
	
	if crop == False & resize == True & RGB == True:
		img_crop = tf.random_crop(img_float, size=[135, 480, 3])
		img_resize = tf.image.resize_images(img_crop, size=[70, 240])
		label = tf.one_hot(label, NUM_CLASSES)

		return img_crop, label

	if crop == False & resize == False & RGB == True:
		img_crop = tf.random_crop(img_float, size=[135, 480, 3])
		label = tf.one_hot(label, NUM_CLASSES)

		return img_crop, label
		
	if crop == False & resize == True & RGB == False:
		img_crop = tf.random_crop(img_float, size=[135, 480,1])
		img_resize = tf.image.resize_images(img_crop, size=[70, 240])
		label = tf.one_hot(label, NUM_CLASSES)

		return img_crop, label

	if crop == False & resize == False & RGB == False:
		img_crop = tf.random_crop(img_float, size=[135, 480,1])
		label = tf.one_hot(label, NUM_CLASSES)

		return img_crop, label

		

#def input_tensor_regression(img_path, label):
#	img_file = tf.read_file(img_path)
#	img_decoded = tf.image.decode_png(img_file)
#	img_crop = tf.image.crop_to_bounding_box(img_decoded, 135, 0, 135, 480)
#	img_float = tf.to_float(img_crop)
#	img_crop = tf.random_crop(img_float, size=[135, 135, 3])
#	label = tf.cast(label, tf.float32)
#
#	return img_crop, label



def make_batch(dataset):
    dataset_0 = dataset.filter(
        lambda x, y: tf.reshape(tf.equal(tf.argmax(y), tf.argmax(tf.constant([1, 0], tf.float32))), []))
    dataset_1 = dataset.filter(
        lambda x, y: tf.reshape(tf.equal(tf.argmax(y), tf.argmax(tf.constant([0, 1], tf.float32))), [])).repeat()

    datasets = tf.data.Dataset.zip((dataset_0, dataset_1))
    datasets = datasets.flat_map(
        lambda ex_0, ex_1: tf.data.Dataset.from_tensors(ex_0).concatenate(tf.data.Dataset.from_tensors(ex_1)))

    return datasets


def conv2d(x, num_outputs, batch_norm=True):
	if batch_norm is True:
		conv_bn = tf.contrib.layers.batch_norm
	else:
		conv_bn = None

	conv = tf.contrib.layers.conv2d(inputs=x,
									num_outputs=num_outputs,
									kernel_size=(3, 3),
									normalizer_fn=conv_bn,
									activation_fn=tf.nn.relu)
	return conv


def pooling(x):
	pool = tf.contrib.layers.max_pool2d(inputs=x, kernel_size=(2, 2))
	return pool


def dense(x, output, fn=tf.nn.relu, batch_norm=True):
	if batch_norm is True:
		fc_bn = tf.contrib.layers.batch_norm
	else:
		fc_bn = None
	fc = tf.contrib.layers.fully_connected(inputs=x,
	                                       num_outputs=output,
	                                       normalizer_fn=fc_bn,
	                                       activation_fn=fn)
	return fc


class VGG16():
    def __init__(self, x, y, bn, classification):

        with tf.name_scope("input"):
            self.x = x
            self.y = y

        with tf.name_scope("layer_1"):
            conv1 = conv2d(x, 16, batch_norm=bn)
            conv2 = conv2d(conv1, 16, batch_norm=bn)
            pool1 = pooling(conv2)

        with tf.name_scope("layer_2"):
            conv3 = conv2d(pool1, 32, batch_norm=bn)
            conv4 = conv2d(conv3, 32, batch_norm=bn)
            pool2 = pooling(conv4)

        with tf.name_scope("layer_3"):
            conv5 = conv2d(pool2, 64, batch_norm=bn)
            conv6 = conv2d(conv5, 64, batch_norm=bn)
            conv7 = conv2d(conv6, 64, batch_norm=bn)
            pool3 = pooling(conv7)

        with tf.name_scope("layer_4"):
            conv8 = conv2d(pool3, 128, batch_norm=bn)
            conv9 = conv2d(conv8, 128, batch_norm=bn)
            conv10 = conv2d(conv9, 128, batch_norm=bn)
            pool4 = pooling(conv10)

        with tf.name_scope("layer_5"):
            conv11 = conv2d(pool4, 128, batch_norm=bn)
            conv12 = conv2d(conv11, 128, batch_norm=bn)
            conv13 = conv2d(conv12, 128, batch_norm=bn)
            pool5 = pooling(conv13)

        with tf.name_scope("FC_layer"):
            fc1 = tf.layers.flatten(pool5)
            fc2 = dense(fc1, 4096, batch_norm=bn)
            fc3 = dense(fc2, 4096, batch_norm=bn)

        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if classification is True:
            self.logits = dense(fc3, NUM_CLASSES, fn=None, batch_norm=True)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)

            self.y_prob = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.y_prob, 1)

            self.correct_prediction = tf.equal(self.y_pred, tf.arg_max(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            tf.summary.scalar("accuray", self.accuracy)
            tf.summary.scalar("loss", self.loss)

        else:
            self.logits = tf.layers.dense(fc3, 1, activation=tf.nn.relu)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)

            tf.summary.scalar("loss", self.loss)

        self.merged_summary_op = tf.summary.merge_all()



class Inception_v3():
    def __init__(self, x, y, bn, classification):
        
        self.x = x
        self.y = y
        
        conv1_7_7 = conv_2d(x, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
        
        # 3a
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3,  activation='relu', name='inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)



        # 3b
        inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name='inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu', name='inception_3b_pool_1_1')
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
        
        # 4a
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

        # 4b
        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

        # 4c
        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')

        # 4d
        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

        # 4e
        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3, mode='concat')
        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

        # 5a
        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu', name='inception_5a_pool_1_1')
        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')

        # 5b
        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3, activation='relu', name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)
        
        fc1 = tf.layers.flatten(pool5_7_7)
     

        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if classification is True:
            self.logits = dense(fc1, NUM_CLASSES, fn=None, batch_norm=True)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)

            self.y_prob = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.y_prob, 1)

            self.correct_prediction = tf.equal(self.y_pred, tf.arg_max(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            tf.summary.scalar("accuray", self.accuracy)
            tf.summary.scalar("loss", self.loss)

        else:
            self.logits = tf.layers.dense(fc1, 1, activation=tf.nn.relu)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)
            
            tf.summary.scalar("loss", self.loss)

        self.merged_summary_op = tf.summary.merge_all()
		
		
		
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out

def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    return res
	
class Resnet():
    def __init__(self, x, y, bn, classification, n):
        
        if n < 20 or (n - 20) % 12 != 0:
            print ("ResNet depth invalid.")
            return
        
        num_conv = (n - 20) / 12 + 1
        num_conv = int(num_conv)
        layers = []
        
        with tf.name_scope("input"):
            self.x = x
            self.y = y

            
        with tf.variable_scope('conv1'):
            conv1 = conv_layer(x, [3,3,3,16],1)
            layers.append(conv1)
        
        for i in range (num_conv):
            with tf.variable_scope('conv2_%d' % (i+1)):
                conv2_x = residual_block(layers[-1], 16, False)
                conv2 = residual_block(conv2_x, 16, False)
                layers.append(conv2_x)
                layers.append(conv2)


        for i in range (num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv3_%d' % (i+1)):
                conv3_x = residual_block(layers[-1], 32, down_sample)
                conv3 = residual_block(conv3_x, 32, False)
                layers.append(conv3_x)
                layers.append(conv3)


        for i in range (num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv4_%d' % (i+1)):
                conv4_x = residual_block(layers[-1], 64, down_sample)
                conv4 = residual_block(conv4_x, 64, False)
                layers.append(conv4_x)
                layers.append(conv4)

            
        
        global_pool = tf.reduce_mean(layers[-1],[1,2])
        assert global_pool.get_shape().as_list()[1:] == [64]
        
        out = softmax_layer(global_pool, [64, NUM_CLASSES])
        layers.append(out)
        
        
        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if classification is True:
            #self.logits = dense(fc1, NUM_CLASSES, fn=None, batch_norm=True)
            self.logits = layers[-1]
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)

            self.y_prob = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.y_prob, 1)

            self.correct_prediction = tf.equal(self.y_pred, tf.arg_max(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            tf.summary.scalar("accuray", self.accuracy)
            tf.summary.scalar("loss", self.loss)

        else:
            self.logits = tf.layers.dense(fc1, 1, activation=tf.nn.relu)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)
            
            tf.summary.scalar("loss", self.loss)

        self.merged_summary_op = tf.summary.merge_all()
		
		
		
class Densenet():
    def __init__(self, x, y, bn, classification, k,L):
        
        nb_layers = int((L - 4) / 3)
        
        with tf.name_scope("input"):
            self.x = x
            self.y = y

        with tf.name_scope("layer_1"):
            net = tflearn.conv_2d(x, 16, 3, regularizer='L2', weight_decay=0.0001)
            net = tflearn.layers.conv.densenet_block(net, nb_layers, k)
            net = tflearn.layers.conv.densenet_block(net, nb_layers, k)
            net = tflearn.layers.conv.densenet_block(net, nb_layers, k)
            net = tflearn.global_avg_pool(net)
            fc1 = tf.layers.flatten(net)

        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if classification is True:
            self.logits = dense(fc1, NUM_CLASSES, fn=None, batch_norm=True)
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)

            self.y_prob = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.y_prob, 1)

            self.correct_prediction = tf.equal(self.y_pred, tf.arg_max(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            tf.summary.scalar("accuray", self.accuracy)
            tf.summary.scalar("loss", self.loss)

        else:
            self.logits = tf.layers.dense(fc1, 1, activation=tf.nn.relu)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)
            self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(self.extra_update_ops):
                self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                           global_step=self.global_step)
                self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                     global_step=self.global_step)
                self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                             global_step=self.global_step)
                self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                 global_step=self.global_step)
            
            tf.summary.scalar("loss", self.loss)

        self.merged_summary_op = tf.summary.merge_all()
