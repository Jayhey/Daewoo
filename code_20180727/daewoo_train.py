import pandas as pd
import os
import tensorflow as tf
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
crop = False # True = 10 crop, False = original size
RGB = False # True = RGB, False = Grayscale
resize = True # True = 240x70, False = 480x135 

if crop == True:
	batch_size = 64
else:
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

if classification == True & crop == True:
    train_imgs = train_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor).apply(tf.contrib.data.unbatch()).batch(batch_size)

if classification == True & crop == False:
    train_imgs = train_imgs.map(input_tensor).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    test_imgs = test_imgs.map(input_tensor).shuffle(buffer_size=100).apply(lambda x: make_batch(x)).batch(batch_size).repeat()
    infer_imgs = infer_imgs.map(input_tensor).batch(batch_size)

	

train_iterator = train_imgs.make_initializable_iterator()
test_iterator = test_imgs.make_initializable_iterator()
infer_iterator = infer_imgs.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_imgs.output_types, train_imgs.output_shapes)
x, y = iterator.get_next()

# train class: [22789, 19659]
if crop == True:
	train_batches = 22789*2*10 // batch_size
else:
	train_batches = 22789*2 // batch_size

######################
##### 4개 모델 중 선택 ###
######################

model = VGG16(x, y, bn=True, classification=classification)
model = Inception_v3(x, y, bn=True, classification=classification)
model = Resnet(x, y, bn=True, classification=classification, n=56)
model = Densenet(x, y, bn=True, classification=classification, k=12, L=40)

# 모델 이름 설정하기
if classification is True:
    model_name = "{model_name}_classification_{Crop O, X}_{RGB O, X}_{resize O, X}"
else:
    model_name = "{model_name}_regression_{Crop O, X}_{RGB O, X}_{resize O, X}"

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



#####################
# Training###########
#####################
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

    saver.save(sess, os.path.join(logs_path, '{model_name}_classification_{Crop O, X}_{RGB O, X}_{resize O, X}', model_name))

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

    saver.save(sess, os.path.join(logs_path, '{model_name}_regression_{Crop O, X}_{RGB O, X}_{resize O, X}', model_name))

####################
#Inference##########
####################

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

if crop == True:
	y_true_final = [y_true[10*i] for i in range(len(ts_idx))]
	y_pred_final = [np.argmax([list(y_pred[10*i:10*(i+1)]).count(0),5]) for i in range(len(ts_idx))]

	df2 = pd.DataFrame(data={'y_true':y_true_final, 'y_pred':y_pred_final} )
	df2.to_csv("{}_pred.csv".format(model_name), encoding='utf-8', index=False)


	cm = skm.confusion_matrix(y_true_final, y_pred_final)
	acc = skm.accuracy_score(y_true_final, y_pred_final)  # Accuracy
	print("Accuracy : {}".format(acc))

	pd.DataFrame(cm).to_csv("{}_cm.csv".format(model_name), encoding='utf-8')

	report = skm.precision_recall_fscore_support(y_true_final, y_pred_final)
	specificity_0 = cm[1][1]/(cm[1][1]+cm[1][0])
	out_dict = { "precision" :report[0].round(3), "recall" : report[1].round(3),"f1-score" : report[2].round(3),
				 "BCR": np.sqrt(report[1][0]*specificity_0).round(3)}

	pd.DataFrame(out_dict).to_csv("{}_report.csv".format(model_name), encoding='utf-8')

if crop == False:
	df2 = pd.DataFrame(data={'y_true':y_true, 'y_pred':y_pred} )
	df2.to_csv("{}_pred.csv".format(model_name), encoding='utf-8', index=False)


	cm = skm.confusion_matrix(y_true, y_pred)
	acc = skm.accuracy_score(y_true, y_pred)  # Accuracy
	print("Accuracy : {}".format(acc))

	pd.DataFrame(cm).to_csv("{}_cm.csv".format(model_name), encoding='utf-8')

	report = skm.precision_recall_fscore_support(y_true, y_pred)
	specificity_0 = cm[1][1]/(cm[1][1]+cm[1][0])
	out_dict = { "precision" :report[0].round(3), "recall" : report[1].round(3),"f1-score" : report[2].round(3),
				 "BCR": np.sqrt(report[1][0]*specificity_0).round(3)}

	pd.DataFrame(out_dict).to_csv("{}_report.csv".format(model_name), encoding='utf-8')

