##### 패키지 불러오기
import pandas as pd
import os
import tensorflow as tf
import sklearn.metrics as skm
import time
from collections import Counter
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

##### 이미지 전처리, 네트워크 함수
NUM_CLASSES = 2

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
def input_tensor(img_path, label):
    label_crop = tf.one_hot(label, NUM_CLASSES)
    
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_file)
    img_crop = tf.image.crop_to_bounding_box(img_decoded, 135, 0, 135, 135)
    img_float = tf.to_float(img_crop)
    img_crop = tf.random_crop(img_float, size=[135, 135, 3])
    
    for i in range(1,10):
        _label = tf.one_hot(label, NUM_CLASSES)
        
        _img_crop = tf.image.crop_to_bounding_box(img_decoded, 135, 38*i, 135, 135)
        _img_float = tf.to_float(_img_crop)
        _img_crop = tf.random_crop(_img_float, size=[135, 135, 3])
        
        label_crop = tf.concat([label_crop, _label], axis=0)
        img_crop = tf.concat([img_crop, _img_crop], axis=0)

    return tf.reshape(img_crop, [-1,135,135,3]), tf.reshape(label_crop, [-1,NUM_CLASSES])
	
	
def make_batch(dataset):
    dataset_0 = dataset.filter(lambda x,y: tf.reshape(tf.equal(tf.argmax(y), tf.argmax(tf.constant([1,0], tf.float32))), []))
    dataset_1 = dataset.filter(lambda x,y: tf.reshape(tf.equal(tf.argmax(y), tf.argmax(tf.constant([0,1], tf.float32))), [])).repeat()
    
    datasets = tf.data.Dataset.zip((dataset_0, dataset_1))
    datasets = datasets.flat_map(lambda ex_0, ex_1: tf.data.Dataset.from_tensors(ex_0).concatenate(tf.data.Dataset.from_tensors(ex_1)))
    
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
		
		
	
###### 데이터 불러오기, 학습하기
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

root_dir = ".\\input_data"
img_dir = "./input_data/figure/"
logs_path = os.path.join(root_dir, "graph")

df = pd.read_csv(os.path.join(root_dir, 'description.csv'), engine='python')
img = df.img_name.values
img = np.array([img_dir + x for x in img])

classification = True

batch_size = 64
epochs = 5

if classification is True:
   label = pd.cut(df['WVHT ft.y'], bins=[0, 5.2, 7.9, 100], labels=[0, 1, 2], include_lowest=True).values
else:
   label = df['WVHT ft.y'].values
   label = ((label - np.mean(label)) / np.std(label)).reshape(-1, 1)
   
idx = [i for i in range(len(label)) if label[i] !=1]

img = img[idx]
label = label[idx]


for i in range(len(label)):
    if label[i] == 2:
        label[i] = 1
		
		
# Tensorflow Dataset API
train_img_tensor, train_label_tensor, test_img_tensor, test_label_tensor, tr_idx, ts_idx = set_input(img, label)

train_imgs = tf.data.Dataset.from_tensor_slices((train_img_tensor, train_label_tensor))
test_imgs = tf.data.Dataset.from_tensor_slices((test_img_tensor, test_label_tensor))
infer_imgs = tf.data.Dataset.from_tensor_slices((test_img_tensor, test_label_tensor))

if classification is True:
    train_imgs = train_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).batch(batch_size)
else:
    train_imgs = train_imgs.map(input_tensor_regression).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_regression).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).batch(batch_size)

train_iterator = train_imgs.make_initializable_iterator()
test_iterator = test_imgs.make_initializable_iterator()
infer_iterator = infer_imgs.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_imgs.output_types, train_imgs.output_shapes)
x, y = iterator.get_next()

# train class: [22789, 19659]
train_batches = 22789*2*10 // batch_size

model = Inception_v3(x, y, bn=True, classification=classification)

if classification is True:
    model_name = "Googlenet_classification_crop10_voting"
else:
    model_name = "Googlenet_regression_crop10_voting"

start_time = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
train_handle = sess.run(train_iterator.string_handle())
test_handle = sess.run(test_iterator.string_handle())
infer_handle = sess.run(infer_iterator.string_handle())
train_writer = tf.summary.FileWriter(os.path.join(logs_path, model_name, 'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(logs_path, model_name, 'test'))

LEARNING_RATE = 0.001
optimizer = model.rms

# Training

if classification is True:

    print("Training!")
    for i in range(epochs):
        print("-------{} Epoch--------".format(i + 1))
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        for j in range(train_batches):
            summary, _, acc, loss_ = sess.run([model.merged_summary_op, optimizer, model.accuracy, model.loss],
                                              feed_dict={handle: train_handle, model.learning_rate: LEARNING_RATE})
            step = tf.train.global_step(sess, model.global_step)
            print("Training Iter : {}, Acc : {}, Loss : {:.4f}".format(step, acc, loss_))

            if j % 10 == 0:
                train_writer.add_summary(summary, step)
                summary, acc, loss_ = sess.run([model.merged_summary_op, model.accuracy, model.loss],
                                               feed_dict={handle: test_handle})
                print("Validation Iter : {}, Acc : {}, Loss : {:.4f}".format(step, acc, loss_))
                test_writer.add_summary(summary, step)

    print("-----------End of training-------------")

    end_time = time.time() - start_time
    print("{} seconds".format(end_time))

    saver.save(sess, os.path.join(logs_path, 'Googlenet_classification_crop', model_name))

else:
    print("Training!")
    for i in range(epochs):
        print("-------{} Epoch--------".format(i + 1))
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        for j in range(train_batches):
            summary, _, loss_ = sess.run([model.merged_summary_op, optimizer, model.loss],
                                         feed_dict={handle: train_handle, model.learning_rate: LEARNING_RATE})
            step = tf.train.global_step(sess, model.global_step)
            print("Training Iter : {}, Loss : {:.4f}".format(step, loss_))

            if j % 10 == 0:
                train_writer.add_summary(summary, step)
                summary, loss_ = sess.run([model.merged_summary_op, model.loss],
                                          feed_dict={handle: test_handle})
                print("Validation Iter : {}, Loss : {:.4f}".format(step, loss_))
                test_writer.add_summary(summary, step)

    print("-----------End of training-------------")

    end_time = time.time() - start_time
    print("{} seconds".format(end_time))

    saver.save(sess, os.path.join(logs_path, 'Googlenet_regression_crop', model_name))

	
# Inference

sess.run(infer_iterator.initializer)
y_true, y_pred = sess.run([model.y, model.y_pred], feed_dict={handle:infer_handle})
i = 0

 
while True:
    try:
         tmp_true, tmp_pred = sess.run([model.y, model.y_pred], feed_dict={handle:infer_handle})
         y_true = np.concatenate((y_true, tmp_true))
         y_pred = np.concatenate((y_pred, tmp_pred))
         if i % 200 == 0:
             print(i)
         i += 1
    except:
         y_true = np.array([np.where(r==1)[0][0] for r in y_true])
         break

len(y_pred)

y_true_final = [y_true[10*i] for i in range(len(ts_idx))]
y_pred_final = [np.argmax([list(y_pred[10*i:10*(i+1)]).count(0),5]) for i in range(len(ts_idx))]

df2 = pd.DataFrame(data={'y_true':y_true_final, 'y_pred':y_pred_final} )
df2.to_csv("{}_pred.csv".format(model_name), encoding='utf-8', index=False)


cm = skm.confusion_matrix(y_true_final, y_pred_final)
acc = skm.accuracy_score(y_true_final, y_pred_final)  # Accuracy
print("Accuracy : {}".format(acc))

pd.DataFrame(cm).to_csv("{}_cm.csv".format(model_name), encoding='utf-8')

report = skm.precision_recall_fscore_support(y_true_final, y_pred_final)
out_dict = { "precision" :report[0].round(3), "recall" : report[1].round(3),"f1-score" : report[2].round(3),
             "BCR": np.sqrt(report[0]*report[1]).round(3)}

pd.DataFrame(out_dict).to_csv("{}_report.csv".format(model_name), encoding='utf-8')
