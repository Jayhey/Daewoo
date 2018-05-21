import tensorflow as tf


def input_tensor(img_path, label):
	img_file = tf.read_file(img_path)
	img_decoded = tf.image.decode_png(img_file)
	img_float = tf.to_float(img_decoded)
	img_crop = tf.random_crop(img_float, size=[270, 270, 3])
	label = tf.cast(label, tf.float32)
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


class VGG16():
	'''
	VGG 16Network
	'''
	def __init__(self, bn):

		# self.is_regression = regression

		with tf.name_scope("layer_1"):
			conv1 = conv2d(self.x, 64, batch_norm=bn)
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
			fc2 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu)
			fc3 = tf.layers.dense(fc2, 4096, activation=tf.nn.relu)


		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.learning_rate = tf.placeholder(tf.float32)
		self.logits = tf.layers.dense(fc3, 1, activation=tf.nn.relu)
		self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
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



