##### 패키지 불러오기
import pandas as pd
import os
import tensorflow as tf
import sklearn.metrics as skm
import time
from collections import Counter
import tflearn
import tflearn.activations as activations
from tflearn.activations import relu
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.utils import repeat
import numpy as np

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
    img_crop = tf.image.crop_to_bounding_box(img_decoded, 135, 0, 135, 480)
    img_float = tf.to_float(img_crop)
    img_crop = tf.random_crop(img_float, size=[135, 480, 3])
    label = tf.one_hot(label, NUM_CLASSES)
    

    return img_crop, label	
	
	
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
	
def block35(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None,name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 32, 3, bias=False, activation=None,name='Conv2d_0b_3x3')))
    tower_conv2_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 48,3, bias=False, activation=None, name='Conv2d_0b_3x3')))
    tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 64,3, bias=False, activation=None, name='Conv2d_0c_3x3')))
    tower_mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net

def block17(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv_1_0 = relu(batch_normalization(conv_2d(net, 128, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv_1_1 = relu(batch_normalization(conv_2d(tower_conv_1_0, 160,[1,7], bias=False, activation=None,name='Conv2d_0b_1x7')))
    tower_conv_1_2 = relu(batch_normalization(conv_2d(tower_conv_1_1, 192, [7,1], bias=False, activation=None,name='Conv2d_0c_7x1')))
    tower_mixed = merge([tower_conv,tower_conv_1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net


def block8(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 224, [1,3], bias=False, activation=None, name='Conv2d_0b_1x3')))
    tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 256, [3,1], bias=False, name='Conv2d_0c_3x1')))
    tower_mixed = merge([tower_conv,tower_conv1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net
	
dropout_keep_prob = 0.8

class Inception_v4():
    def __init__(self, x, y, bn, classification):
        
        self.x = x
        self.y = y
        
        conv1a_3_3 = relu(batch_normalization(conv_2d(x, 32, 3, strides=2, bias=False, padding='VALID',activation=None,name='Conv2d_1a_3x3')))
        conv2a_3_3 = relu(batch_normalization(conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID',activation=None, name='Conv2d_2a_3x3')))
        conv2b_3_3 = relu(batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation=None, name='Conv2d_2b_3x3')))
        maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
        conv3b_1_1 = relu(batch_normalization(conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID',activation=None, name='Conv2d_3b_1x1')))
        conv4a_3_3 = relu(batch_normalization(conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID',activation=None, name='Conv2d_4a_3x3')))
        maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')

        tower_conv = relu(batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation=None, name='Conv2d_5b_b0_1x1')))

        tower_conv1_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation=None, name='Conv2d_5b_b1_0a_1x1')))
        tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation=None, name='Conv2d_5b_b1_0b_5x5')))

        tower_conv2_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation=None, name='Conv2d_5b_b2_0a_1x1')))
        tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0b_3x3')))
        tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation=None,name='Conv2d_5b_b2_0c_3x3')))

        tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
        tower_conv3_1 = relu(batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation=None,name='Conv2d_5b_b3_0b_1x1')))

        tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)

        net = repeat(tower_5b_out, 10, block35, scale=0.17)

        tower_conv = relu(batch_normalization(conv_2d(net, 384, 3, bias=False, strides=2,activation=None, padding='VALID', name='Conv2d_6a_b0_0a_3x3')))
        tower_conv1_0 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_6a_b1_0a_1x1')))
        tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 256, 3, bias=False, activation=None, name='Conv2d_6a_b1_0b_3x3')))
        tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 384, 3, bias=False, strides=2, padding='VALID', activation=None,name='Conv2d_6a_b1_0c_3x3')))
        tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID',name='MaxPool_1a_3x3')
        net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
        net = repeat(net, 20, block17, scale=0.1)

        tower_conv = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
        tower_conv0_1 = relu(batch_normalization(conv_2d(tower_conv, 384, 3, bias=False, strides=2, padding='VALID', activation=None,name='Conv2d_0a_1x1')))

        tower_conv1 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, padding='VALID', activation=None,name='Conv2d_0a_1x1')))
        tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1,288,3, bias=False, strides=2, padding='VALID',activation=None, name='COnv2d_1a_3x3')))

        tower_conv2 = relu(batch_normalization(conv_2d(net, 256,1, bias=False, activation=None,name='Conv2d_0a_1x1')))
        tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2, 288,3, bias=False, name='Conv2d_0b_3x3',activation=None)))
        tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 320, 3, bias=False, strides=2, padding='VALID',activation=None, name='Conv2d_1a_3x3')))

        tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
        net = merge([tower_conv0_1, tower_conv1_1,tower_conv2_2, tower_pool], mode='concat', axis=3)

        net = repeat(net, 9, block8, scale=0.2)
        net = block8(net, activation=None)

        net = relu(batch_normalization(conv_2d(net, 1536, 1, bias=False, activation=None, name='Conv2d_7b_1x1')))
        net = avg_pool_2d(net, net.get_shape().as_list()[1:3],strides=2, padding='VALID', name='AvgPool_1a_8x8')
        net = flatten(net)
        net = dropout(net, dropout_keep_prob)
        
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


###### 데이터 불러오기, 학습하기
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

root_dir = ".\\input_data"
img_dir = "./input_data/figure/"
logs_path = os.path.join(root_dir, "graph")

df = pd.read_csv(os.path.join(root_dir, 'description.csv'), engine='python')
img = df.img_name.values
img = np.array([img_dir + x for x in img])

classification = True

batch_size = 16
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
    train_imgs = train_imgs.map(input_tensor).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor).batch(batch_size)
else:
    train_imgs = train_imgs.map(input_tensor_regression).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_regression).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor).batch(batch_size)

train_iterator = train_imgs.make_initializable_iterator()
test_iterator = test_imgs.make_initializable_iterator()
infer_iterator = infer_imgs.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_imgs.output_types, train_imgs.output_shapes)
x, y = iterator.get_next()

# train class: [22789, 19659]
train_batches = 22789*2 // batch_size

model = Inception_v4(x, y, bn=True, classification=classification)

if classification is True:
    model_name = "Inception_v4_classification_cropX"
else:
    model_name = "Inception_v4_regression_cropX"

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

    saver.save(sess, os.path.join(logs_path, 'Inception_v4_classification_crop', model_name))

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

    saver.save(sess, os.path.join(logs_path, 'Inception_v4_regression_crop', model_name))

	
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


df2 = pd.DataFrame(data={'y_true':y_true, 'y_pred':y_pred} )
df2.to_csv("{}_pred.csv".format(model_name), encoding='utf-8', index=False)


cm = skm.confusion_matrix(y_true, y_pred)
acc = skm.accuracy_score(y_true, y_pred)  # Accuracy
print("Accuracy : {}".format(acc))

pd.DataFrame(cm).to_csv("{}_cm.csv".format(model_name), encoding='utf-8')

report = skm.precision_recall_fscore_support(y_true, y_pred)
out_dict = { "precision" :report[0].round(3), "recall" : report[1].round(3),"f1-score" : report[2].round(3),
             "BCR": np.sqrt(report[0]*report[1]).round(3)}

pd.DataFrame(out_dict).to_csv("{}_report.csv".format(model_name), encoding='utf-8')
