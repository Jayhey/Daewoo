import tensorflow as tf
import numpy as np

NUM_CLASSES = 3


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

	return train_img_tensor, train_label_tensor, test_img_tensor, test_label_tensor


# string 텐서를 img 텐서로 변환 후 crop
def input_tensor_regression(img_path, label):
	img_file = tf.read_file(img_path)
	img_decoded = tf.image.decode_png(img_file)
	img_float = tf.to_float(img_decoded)
	img_crop = tf.random_crop(img_float, size=[270, 270, 3])
	label = tf.cast(label, tf.float32)
	return img_crop, label

# string 텐서를 img 텐서로 변환 후 crop (classification)
def input_tensor(img_path, label):
	img_file = tf.read_file(img_path)
	img_decoded = tf.image.decode_png(img_file)
	img_float = tf.to_float(img_decoded)
	img_crop = tf.random_crop(img_float, size=[270, 270, 3])
	label = tf.one_hot(label, NUM_CLASSES)

	return img_crop, label

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

		with tf.name_scope("layer_1"):
			conv1 = conv2d(x, 64, batch_norm=bn)
			conv2 = conv2d(conv1, 64, batch_norm=bn)
			pool1 = pooling(conv2)

		with tf.name_scope("layer_2"):
			conv3 = conv2d(pool1, 128, batch_norm=bn)
			conv4 = conv2d(conv3, 128, batch_norm=bn)
			pool2 = pooling(conv4)

		with tf.name_scope("layer_3"):
			conv5 = conv2d(pool2, 256, batch_norm=bn)
			conv6 = conv2d(conv5, 256, batch_norm=bn)
			conv7 = conv2d(conv6, 256, batch_norm=bn)
			pool3 = pooling(conv7)

		with tf.name_scope("layer_4"):
			conv8 = conv2d(pool3, 512, batch_norm=bn)
			conv9 = conv2d(conv8, 512, batch_norm=bn)
			conv10 = conv2d(conv9, 512, batch_norm=bn)
			pool4 = pooling(conv10)

		with tf.name_scope("layer_5"):
			conv11 = conv2d(pool4, 512, batch_norm=bn)
			conv12 = conv2d(conv11, 512, batch_norm=bn)
			conv13 = conv2d(conv12, 512, batch_norm=bn)
			pool5 = pooling(conv13)

		with tf.name_scope("FC_layer"):
			fc1 = tf.layers.flatten(pool5)
			fc2 = dense(fc1, 4096, batch_norm=bn)
			fc3 = dense(fc2, 4096, batch_norm=bn)

		self.learning_rate = tf.placeholder(tf.float32)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		if classification is True:
			self.logits = dense(fc3, NUM_CLASSES, fn=None, batch_norm=True)
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=self.logits)
			# self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

			self.train = self.optimizer.minimize(self.loss)

			self.y_prob = tf.nn.softmax(self.logits)
			self.y_pred = tf.argmax(self.y_prob, 1)

			self.correct_prediction = tf.equal(self.y_pred, tf.arg_max(y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

			tf.summary.scalar("accuray", self.accuracy)
			tf.summary.scalar("loss", self.loss)

		else:
			self.logits = tf.layers.dense(fc3, 1, activation=tf.nn.relu)
			self.loss = tf.losses.mean_squared_error(labels=y, predictions=self.logits)
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.train = self.optimizer.minimize(self.loss)

			tf.summary.scalar("loss", self.loss)

		self.merged_summary_op = tf.summary.merge_all()







		# with tf.name_scope('train'):
		# 	self.global_step = tf.Variable(0, trainable=False, name='global_step')
		# 	self.learning_rate = tf.placeholder(tf.float32)
		# 	self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# 	with tf.control_dependencies(self.extra_update_ops):
		# 		self.adam = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy,
		# 		                                                                global_step=self.global_step)
		# 		self.sgd = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy,
		# 		                                                                          global_step=self.global_step)
		# 		self.rms = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cross_entropy,
		# 		                                                                  global_step=self.global_step)
		# 		self.momentum = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(
		# 			self.cross_entropy,
		# 			global_step=self.global_step)
		#
		# with tf.name_scope('accuracy'):
		# 	with tf.name_scope('correct_prediction'):
		# 		self.correct_prediction = tf.equal(self.y_pred, self.y_)
		# 	with tf.name_scope('accuracy'):
		# 		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		# 		tf.add_to_collection('summaries_general', tf.summary.scalar('accuracy', self.accuracy))
		#
		#
		# self.merged = tf.summary.merge(tf.get_collection('summaries_general'))
		# self.img_summary = tf.summary.merge(tf.get_collection('summaries_img'))


# def VGG16(x, bn):
#
# 	with tf.name_scope("layer_1"):
# 		conv1 = conv2d(x, 64, batch_norm=bn)
# 		conv2 = conv2d(conv1, 64, batch_norm=bn)
# 		pool1 = pooling(conv2)
#
# 	with tf.name_scope("layer_2"):
# 		conv3 = conv2d(pool1, 128, batch_norm=bn)
# 		conv4 = conv2d(conv3, 128, batch_norm=bn)
# 		pool2 = pooling(conv4)
#
# 	with tf.name_scope("layer_3"):
# 		conv5 = conv2d(pool2, 256, batch_norm=bn)
# 		conv6 = conv2d(conv5, 256, batch_norm=bn)
# 		conv7 = conv2d(conv6, 256, batch_norm=bn)
# 		pool3 = pooling(conv7)
#
# 	with tf.name_scope("layer_4"):
# 		conv8 = conv2d(pool3, 512, batch_norm=bn)
# 		conv9 = conv2d(conv8, 512, batch_norm=bn)
# 		conv10 = conv2d(conv9, 512, batch_norm=bn)
# 		pool4 = pooling(conv10)
#
# 	with tf.name_scope("layer_5"):
# 		conv11 = conv2d(pool4, 512, batch_norm=bn)
# 		conv12 = conv2d(conv11, 512, batch_norm=bn)
# 		conv13 = conv2d(conv12, 512, batch_norm=bn)
# 		pool5 = pooling(conv13)
#
# 	with tf.name_scope("FC_layer"):
# 		fc1 = tf.layers.flatten(pool5)
# 		fc2 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu)
# 		fc3 = tf.layers.dense(fc2, 4096, activation=tf.nn.relu)
#
# 	global_step = tf.Variable(0, trainable=False, name='global_step')
# 	learning_rate = tf.placeholder(tf.float32)
# 	logits = tf.layers.dense(fc3, 1, activation=tf.nn.relu)
# 	loss = tf.losses.mean_squared_error(labels=y, predictions=logits)
# 	optimizer = tf.train.RMSPropOptimizer(learning_rate)
# 	train = optimizer.minimize(loss)
# 	tf.summary.scalar("loss", loss)
# 	merged_summary_op = tf.summary.merge_all()

