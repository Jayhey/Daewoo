import pandas as pd
import os
import tensorflow as tf
#import os
#os.chdir('D:/daewoo')
#import sys
#sys.path.append('D:/daewoo')


from daewoo_module import *
import sklearn.metrics as skm
import time
from collections import Counter
import numpy as np
import tflearn


###### 데이터 불러오기, 학습하기
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

root_dir = ".\\input_data"
img_dir = "./input_data/figure/"
logs_path = os.path.join(root_dir, "graph")

df = pd.read_csv(os.path.join(root_dir, 'description.csv'), engine='python')
img = df.img_name.values
img = np.array([img_dir + x for x in img])

########################
##### 반드시 확인 필요#######
########################
classification = True
crop = False  # True = 10 crop, False = original size
RGB = False  # True = RGB, False = Grayscale
resize = True # True = 240x70, False = 480x135


batch_size = 64
epochs = 101

if classification is True:
    label = pd.cut(df['WVHT ft.y'], bins=[0, 5.2, 7.9, 100], labels=[0, 1, 2], include_lowest=True).values
else:
    label = df['WVHT ft.y'].values
    label = ((label - np.mean(label)) / np.std(label)).reshape(-1, 1)

idx = [i for i in range(len(label)) if label[i] != 1]

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

'''
if classification == True and crop == True and RGB == True:
    train_imgs = train_imgs.map(input_tensor_crop_RGB).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_crop_RGB).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor_crop_RGB).apply(tf.contrib.data.unbatch()).batch(batch_size)


if classification == True and crop == True and RGB == False:
    train_imgs = train_imgs.map(input_tensor_crop_gray).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(
        lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_crop_gray).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(
        lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor_crop_gray).apply(tf.contrib.data.unbatch()).batch(batch_size)


if classification == True and crop == False and resize == False and RGB == True:
    train_imgs = train_imgs.map(input_tensor_resizeX_RGB).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_resizeX_RGB).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor_resizeX_RGB).batch(batch_size)

if classification == True and crop == False and resize == False and RGB == False:
    train_imgs = train_imgs.map(input_tensor_resizeX_gray).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_resizeX_gray).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor_resizeX_gray).batch(batch_size)

if classification == True and crop == False and resize == True and RGB == True:
    train_imgs = train_imgs.map(input_tensor_resizeO_RGB).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_resizeO_RGB).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor_resizeO_RGB).batch(batch_size)

'''


if classification == True and crop == False and resize == True and RGB == False:
    train_imgs = train_imgs.map(input_tensor_resizeO_gray).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor_resizeO_gray).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(
        batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor_resizeO_gray).batch(batch_size)



train_iterator = train_imgs.make_initializable_iterator()
test_iterator = test_imgs.make_initializable_iterator()
infer_iterator = infer_imgs.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_imgs.output_types, train_imgs.output_shapes)
x, y = iterator.get_next()

# train class: [22789, 19659]
if crop == True:
    train_batches = 22789 * 2 * 10 // batch_size
else:
    train_batches = 22789 * 2 // batch_size
	
######################
##### 4개 모델 중 선택 ###
######################

model = VGG16_CONCAT(x, y, bn=True, classification=classification)
#model = Inception_v3(x, y, bn=True, classification=classification)
#model = Resnet(x, y, bn=True, classification=classification, n=56)
#model = Densenet(x, y, bn=True, classification=classification, k=12, L=40)

# 모델 이름 설정하기
model_name = "{model_name}_regression_{Crop O, X}_{RGB O, X}_{resize O, X}"

start_time = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=110)
sess.run(tf.global_variables_initializer())
train_handle = sess.run(train_iterator.string_handle())
test_handle = sess.run(test_iterator.string_handle())
infer_handle = sess.run(infer_iterator.string_handle())
train_writer = tf.summary.FileWriter(os.path.join(logs_path, model_name, 'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(logs_path, model_name, 'test'))

LEARNING_RATE = 0.001
optimizer = model.rms


#####################
# Training###########
#####################
start_time = time.time()

earlystop_threshold = 30

if not os.path.exists('./checkpoints_concat_ensemble'):
    os.mkdir('./checkpoints_concat_ensemble')

f = open('./concat_ensemble_val_score.csv', 'w')

f.write('epoch,step,val_acc,val_loss,time\n')


if classification is True:

    print("Training!")
    for i in range(epochs):     
        print("-------{} Epoch--------".format(i + 1))
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)

        # LEARNING_RATE *= 0.93
        flag = True
        earlystop_cnt = 0
        max_val_acc = 0.0
        for j in range(train_batches):
            summary, _, acc, loss_ = sess.run([model.merged_summary_op, optimizer, model.accuracy, model.loss],
                                              feed_dict={handle: train_handle, model.learning_rate: LEARNING_RATE})
            step = tf.train.global_step(sess, model.global_step)

            if i==0:
                print(j+1)

            # Early stop and save.
            if i != 0 and flag:
                print("Training Iter : {}, Acc : {}, Loss : {:.4f}".format(step, acc, loss_))

                cur_val_acc, cur_val_loss = sess.run([model.accuracy, model.loss], feed_dict={handle:test_handle})
                if cur_val_acc < max_val_acc:
                    if earlystop_cnt == earlystop_threshold:
                        print('Early saved on step {}'.format(step))
                        flag = False
                    else:
                        print('Overfitting warning: {}'.format(earlystop_cnt))
                        earlystop_cnt += 1
                else:
                    earlystop_cnt = 0
                    max_val_acc = cur_val_acc
                    saver.save(sess, './checkpoints_concat_ensemble/model{}/model.ckpt'.format(i))
                    print("\tSaved step {}, val_acc : {}, val_loss : {:.4f}".format(step, max_val_acc, cur_val_loss))
                    end_time = time.time() - start_time
                    f.write('{},{},{},{},{}\n'.format(i, step, max_val_acc, cur_val_loss, end_time))

            if flag == False:
                print('Flag False {}'.format(step))

            if i == 100 and flag == False:
                break

    f.close()

#            if j % 10 == 0:
#                train_writer.add_summary(summary, step)
#                summary, acc, loss_ = sess.run([model.merged_summary_op, model.accuracy, model.loss],
#                                               feed_dict={handle: test_handle})
#                print("Validation Iter : {}, Acc : {}, Loss : {:.4f}".format(step, acc, loss_))
#                test_writer.add_summary(summary, step)

    print("-----------End of training-------------")

    end_time = time.time() - start_time
    print("{} seconds".format(end_time))

    saver.save(sess,
               os.path.join(logs_path, model_name))

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

    saver.save(sess,
               os.path.join(logs_path, model_name))
