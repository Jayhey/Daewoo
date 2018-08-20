import pandas as pd
import os
import tensorflow as tf
import os
import sys

from daewoo_module import *
import sklearn.metrics as skm
import time
from collections import Counter
import numpy as np
import tflearn

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

root_dir = "../../"
img_dir = "../../figure/"

df = pd.read_csv(os.path.join(root_dir, 'description.csv'), engine='python')
img = df.img_name.values
img = np.array([img_dir + x for x in img])

classification = True
crop = False  # True = 10 crop, False = original size
RGB = False  # True = RGB, False = Grayscale
resize = True  # True = 240x70, False = 480x135

if crop == True:
    batch_size = 64
else:
    batch_size = 16

epochs = 11

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
    
    
model = VGG16(x, y, bn=True, classification=classification)
model_name='vgg16'

file_list = sorted(os.listdir('./checkpoints/'), key=lambda x: int(x[5:]))

sess = tf.Session()
saver = tf.train.Saver()

os.mkdir('pred_results')
for fn in file_list:
    tf.reset_default_graph()
    saver.restore(sess, './checkpoints/{}/model.ckpt'.format(fn))

    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    infer_handle = sess.run(infer_iterator.string_handle())

    sess.run(infer_iterator.initializer)
    y_true, y_pred = sess.run([model.y, model.y_pred], feed_dict={handle: infer_handle})
    i = 0

    while True:
        try:
            tmp_true, tmp_pred = sess.run([model.y, model.y_pred], feed_dict={handle: infer_handle})
            y_true = np.concatenate((y_true, tmp_true))
            y_pred = np.concatenate((y_pred, tmp_pred))
            i += 1
        except:
            y_true = np.array([np.where(r == 1)[0][0] for r in y_true])
            break

    if crop == False:
        df2 = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
        df2.to_csv("pred_results/{}_pred.csv".format(fn), encoding='utf-8', index=False)

        cm = skm.confusion_matrix(y_true, y_pred)
        acc = skm.accuracy_score(y_true, y_pred)  # Accuracy
        print("Accuracy : {}".format(acc))

        report = skm.precision_recall_fscore_support(y_true, y_pred)
        specificity_0 = cm[1][1] / (cm[1][1] + cm[1][0])
        out_dict = {"precision": report[0].round(3), "recall": report[1].round(3), "f1-score": report[2].round(3),
                    "BCR": np.sqrt(report[1][0] * specificity_0).round(3)}

